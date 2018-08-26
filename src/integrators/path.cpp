
/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

// integrators/path.cpp*
#include "bssrdf.h"
#include "camera.h"
#include "film.h"
#include "integrators/path.h"
#include "interaction.h"
#include "paramset.h"
#include "scene.h"
#include "stats.h"

#include "nanoflann.hpp"
#include "utils.h"

#include <numeric>

#include "hj.h"

namespace pbrt {

#ifdef ADAPTIVE_SAMPLING
typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<Float, PointCloud<Float>>, PointCloud<Float>,
    3 /* dim */
    >
    my_kd_tree_t;
extern std::unique_ptr<my_kd_tree_t> kdtree;
#endif

STAT_PERCENT("Integrator/Zero-radiance paths", zeroRadiancePaths, totalPaths);
STAT_INT_DISTRIBUTION("Integrator/Path length", pathLength);

// PathIntegrator Method Definitions
PathIntegrator::PathIntegrator(int maxDepth,
                               std::shared_ptr<const Camera> camera,
                               std::shared_ptr<Sampler> sampler,
                               const Bounds2i &pixelBounds, Float rrThreshold,
                               const std::string &lightSampleStrategy)
    : SamplerIntegrator(camera, sampler, pixelBounds),
      maxDepth(maxDepth),
      rrThreshold(rrThreshold),
      lightSampleStrategy(lightSampleStrategy) {}

void PathIntegrator::Preprocess(const Scene &scene, Sampler &sampler) {
    lightDistribution =
        CreateLightSampleDistribution(lightSampleStrategy, scene);
}
int counter = 0;
// declare vectors as global variable to avoid memory reallocation for each
// function call
std::vector<std::vector<std::pair<size_t, Float>>> ret_matches(
    MaxThreadIndex());

Spectrum PathIntegrator::Li2(const RayDifferential &r, const Scene &scene,
                             Sampler &sampler, MemoryArena &arena,
                             std::vector<PointInfo> &pointInfoList, int batch,
                             int depth) const {
    ProfilePhase p(Prof::SamplerIntegratorLi);
    Spectrum L(0.f), beta(1.f);
    RayDifferential ray(r);
    bool specularBounce = false;
    int bounces;
    // Added after book publication: etaScale tracks the accumulated effect
    // of radiance scaling due to rays passing through refractive
    // boundaries (see the derivation on p. 527 of the third edition). We
    // track this value in order to remove it from beta when we apply
    // Russian roulette; this is worthwhile, since it lets us sometimes
    // avoid terminating refracted rays that are about to be refracted back
    // out of a medium and thus have their beta value increased.
    Float etaScale = 1;

    PointInfo pInfo;

    for (bounces = 0;; ++bounces) {
        // Find next path vertex and accumulate contribution
        VLOG(2) << "Path tracer bounce " << bounces << ", current L = " << L
                << ", beta = " << beta;

        clock_t localStart, localTime = 0;

        // Intersect _ray_ with scene and store intersection in _isect_
        SurfaceInteraction isect;
        bool foundIntersection = scene.Intersect(ray, &isect);

        // Possibly add emitted light at intersection
        if (bounces == 0 || specularBounce) {
            // Add emitted light at path vertex or from the environment
            if (foundIntersection) {
                L += beta * isect.Le(-ray.d);
                VLOG(2) << "Added Le -> L = " << L;
            } else {
                for (const auto &light : scene.infiniteLights)
                    L += beta * light->Le(ray);
                VLOG(2) << "Added infinite area lights -> L = " << L;
            }
        }

        // Terminate path if ray escaped or _maxDepth_ was reached
        if (!foundIntersection || bounces >= maxDepth) break;

        // Compute scattering functions and skip over medium boundaries
        isect.ComputeScatteringFunctions(ray, arena, true);
        if (!isect.bsdf) {
            VLOG(2) << "Skipping intersection due to null bsdf";
            ray = isect.SpawnRay(ray.d);
            bounces--;
            continue;
        }

        const Distribution1D *distrib = lightDistribution->Lookup(isect.p);

        // Sample illumination from lights to find path contribution.
        // (But skip this for perfectly specular BSDFs.)
        if (isect.bsdf->NumComponents(BxDFType(BSDF_ALL & ~BSDF_SPECULAR)) >
            0) {
            ++totalPaths;
            Spectrum Ld = beta * UniformSampleOneLight(isect, scene, arena,
                                                       sampler, false, distrib);
            VLOG(2) << "Sampled direct lighting Ld = " << Ld;
            if (Ld.IsBlack()) ++zeroRadiancePaths;
            CHECK_GE(Ld.y(), 0.f);
            L += Ld;
        }

        Float cdf[] = {0., 0.25, 0.5, 0.75, 1.0};
        int sampleCounter[4] = {};
        Spectrum sMean[4] = {};
        Spectrum sVariance[4] = {};
        clock_t time[4] = {};
        Float eff[4] = {};
#ifdef ADAPTIVE_SAMPLING
        // std::cout << "batch: " << batch << std::endl;
        if (batch == 1) {
            if (bounces == 0) {
                localStart = clock();
            } else if (bounces == 1) {
                localTime = clock() - localStart;
                if (localTime == 0) counter++;

                pInfo.time = localTime;
                pInfo.radiance = L;

                pointInfoList.push_back(pInfo);
            }

        } else if (batch == 2 && bounces == 1) {
            Float query[] = {ray.o.x, ray.o.y, ray.o.z};

            const Float radius = 1;
            size_t nMatches =
                kdtree->radiusSearch(query, radius, ret_matches[ThreadIndex],
                                     nanoflann::SearchParams());
            if (nMatches == 0) {
                // counter++;
            } else {
                // LOG(INFO) << nMatches << ": " << std::endl;
                std::for_each(ret_matches[ThreadIndex].begin(),
                              ret_matches[ThreadIndex].end(), [&](auto &var) {
                                  auto &pInfo = pointInfoList[var.first];
                                  Float rad = std::atan2f(pInfo.direction.y,
                                                          pInfo.direction.x) /
                                              (2 * Pi);
                                  if (rad < 0) rad += 1.f;
                                  int ind = std::floor(rad / 0.25);
                                  if (ind == 4) ind = 3;

                                  sampleCounter[ind]++;
                                  sMean[ind] += pInfo.radiance;
                                  time[ind] += pInfo.time;
                              });
                for (int i = 0; i < 4; ++i)
                    if (sampleCounter[i] > 0) sMean[i] /= sampleCounter[i];

                std::for_each(
                    ret_matches[ThreadIndex].begin(),
                    ret_matches[ThreadIndex].end(), [&](auto &var) {
                        auto &dir = pointInfoList[var.first].direction;
                        Float rad = std::atan2f(dir.y, dir.x) / (2 * Pi);
                        if (rad < 0) rad += 1.f;
                        int ind = std::floor(rad / 0.25);
                        if (ind == 4) ind = 3;

                        sVariance[ind] +=
                            (pointInfoList[var.first].radiance - sMean[ind]) *
                            (pointInfoList[var.first].radiance - sMean[ind]);
                    });
                for (int i = 0; i < 4; ++i) {
                    if (sampleCounter[i] <= 0) continue;

                    sVariance[i] /= sampleCounter[i];
                    eff[i] =
                        (sVariance[i][0] + sVariance[i][1] + sVariance[i][2]) /
                        3;
                    eff[i] /= std::max(time[i], clock_t(1));
                }

                Float effSum = std::accumulate(eff, eff + 4, 0.f);

                for (int i = 1; i < 5; ++i) {
                    cdf[i] = cdf[i - 1] + eff[i - 1] / effSum;
                }
                /*LOG(INFO) << cdf[1] << ", " << cdf[2] << ", " << cdf[3] << ", "
                          << cdf[4];*/

                /*std::sort(ret_matches.begin(), ret_matches.end(),
                          [&pointInfoList](auto &lhs, auto &rhs) {
                              auto &lhsDir = pointInfoList[lhs.first].direction;
                              auto &rhsDir = pointInfoList[rhs.first].direction;
                              Float lhsRad =
                                  std::atan2f(lhsDir.y, lhsDir.x) / (2 * Pi);
                              lhsRad = lhsRad < 0 ? lhsRad + 1 : lhsRad;
                              Float rhsRad =
                                  std::atan2f(rhsDir.y, rhsDir.x) / (2 * Pi);
                              rhsRad = rhsRad < 0 ? rhsRad + 1 : rhsRad;
                              return lhsRad < rhsRad;
                          });

                clock_t totalTime = std::accumulate(
                    ret_matches[ThreadIndex].begin(),
                    ret_matches[ThreadIndex].end(), clock_t(0),
                    [&pointInfoList](clock_t a, std::pair<size_t, Float> &b) {
                        return a + pointInfoList[b.first].time;
                    });
                Spectrum mean =
                    std::accumulate(
                        ret_matches[ThreadIndex].begin(),
                        ret_matches[ThreadIndex].end(), Spectrum(0.f),
                        [&pointInfoList](Spectrum a,
                                         std::pair<size_t, Float> &b) {
                            return a + pointInfoList[b.first].radiance;
                        }) /
                    nMatches;
                Spectrum variance =
                    std::accumulate(
                        ret_matches[ThreadIndex].begin(),
                        ret_matches[ThreadIndex].end(), Spectrum(0.f),
                        [&pointInfoList, &mean](Spectrum a,
                                                std::pair<size_t, Float> &b) {
                            return a +
                                   (pointInfoList[b.first].radiance - mean) *
                                       (pointInfoList[b.first].radiance - mean);
                        }) /
                    nMatches;

                Float fVariance = (variance[0] + variance[1] + variance[2]) / 3;
                Float eff = fVariance / std::max(clock_t(1), totalTime);*/
            }
        }
#endif

        // Sample BSDF to get new path direction
        Vector3f wo = -ray.d, wi;
        Float pdf;
        BxDFType flags;
        Spectrum f = isect.bsdf->Sample_f2(wo, &wi, sampler.Get2D(), &pdf, cdf,
                                           BSDF_ALL, &flags);

        VLOG(2) << "Sampled BSDF, f = " << f << ", pdf = " << pdf;
        if (f.IsBlack() || pdf == 0.f) break;
        beta *= f * AbsDot(wi, isect.shading.n) / pdf;
        VLOG(2) << "Updated beta = " << beta;
        CHECK_GE(beta.y(), 0.f);
        DCHECK(!std::isinf(beta.y()));
        specularBounce = (flags & BSDF_SPECULAR) != 0;
        if ((flags & BSDF_SPECULAR) && (flags & BSDF_TRANSMISSION)) {
            Float eta = isect.bsdf->eta;
            // Update the term that tracks radiance scaling for refraction
            // depending on whether the ray is entering or leaving the
            // medium.
            etaScale *= (Dot(wo, isect.n) > 0) ? (eta * eta) : 1 / (eta * eta);
        }
        ray = isect.SpawnRay(wi);

#ifdef ADAPTIVE_SAMPLING
        if (batch == 1 && bounces == 0) {
            pInfo.point = ray.o;
            // pInfo.direction = ray.d;
            pInfo.direction = isect.bsdf->WorldToLocal(ray.d);
            // std::cout << pInfo.direction << std::endl;
        }
#endif

        // Account for subsurface scattering, if applicable
        if (isect.bssrdf && (flags & BSDF_TRANSMISSION)) {
            // Importance sample the BSSRDF
            SurfaceInteraction pi;
            Spectrum S = isect.bssrdf->Sample_S(
                scene, sampler.Get1D(), sampler.Get2D(), arena, &pi, &pdf);
            DCHECK(!std::isinf(beta.y()));
            if (S.IsBlack() || pdf == 0) break;
            beta *= S / pdf;

            // Account for the direct subsurface scattering component
            L += beta * UniformSampleOneLight(pi, scene, arena, sampler, false,
                                              lightDistribution->Lookup(pi.p));

            // Account for the indirect subsurface scattering component
            Spectrum f = pi.bsdf->Sample_f(pi.wo, &wi, sampler.Get2D(), &pdf,
                                           BSDF_ALL, &flags);
            if (f.IsBlack() || pdf == 0) break;
            beta *= f * AbsDot(wi, pi.shading.n) / pdf;
            DCHECK(!std::isinf(beta.y()));
            specularBounce = (flags & BSDF_SPECULAR) != 0;
            ray = pi.SpawnRay(wi);
        }

        // Possibly terminate the path with Russian roulette.
        // Factor out radiance scaling due to refraction in rrBeta.
        Spectrum rrBeta = beta * etaScale;
        if (rrBeta.MaxComponentValue() < rrThreshold && bounces > 3) {
            Float q = std::max((Float).05, 1 - rrBeta.MaxComponentValue());
            if (sampler.Get1D() < q) break;
            beta /= 1 - q;
            DCHECK(!std::isinf(beta.y()));
        }
    }
    ReportValue(pathLength, bounces);
    return L;
}

Spectrum PathIntegrator::Li(const RayDifferential &r, const Scene &scene,
                            Sampler &sampler, MemoryArena &arena,
                            int depth) const {
    ProfilePhase p(Prof::SamplerIntegratorLi);
    Spectrum L(0.f), beta(1.f);
    RayDifferential ray(r);
    bool specularBounce = false;
    int bounces;
    // Added after book publication: etaScale tracks the accumulated effect
    // of radiance scaling due to rays passing through refractive
    // boundaries (see the derivation on p. 527 of the third edition). We
    // track this value in order to remove it from beta when we apply
    // Russian roulette; this is worthwhile, since it lets us sometimes
    // avoid terminating refracted rays that are about to be refracted back
    // out of a medium and thus have their beta value increased.
    Float etaScale = 1;

    for (bounces = 0;; ++bounces) {
        // Find next path vertex and accumulate contribution
        VLOG(2) << "Path tracer bounce " << bounces << ", current L = " << L
                << ", beta = " << beta;

        // Intersect _ray_ with scene and store intersection in _isect_
        SurfaceInteraction isect;
        bool foundIntersection = scene.Intersect(ray, &isect);

        // Possibly add emitted light at intersection
        if (bounces == 0 || specularBounce) {
            // Add emitted light at path vertex or from the environment
            if (foundIntersection) {
                L += beta * isect.Le(-ray.d);
                VLOG(2) << "Added Le -> L = " << L;
            } else {
                for (const auto &light : scene.infiniteLights)
                    L += beta * light->Le(ray);
                VLOG(2) << "Added infinite area lights -> L = " << L;
            }
        }

        // Terminate path if ray escaped or _maxDepth_ was reached
        if (!foundIntersection || bounces >= maxDepth) break;

        // Compute scattering functions and skip over medium boundaries
        isect.ComputeScatteringFunctions(ray, arena, true);
        if (!isect.bsdf) {
            VLOG(2) << "Skipping intersection due to null bsdf";
            ray = isect.SpawnRay(ray.d);
            bounces--;
            continue;
        }

        const Distribution1D *distrib = lightDistribution->Lookup(isect.p);

        // Sample illumination from lights to find path contribution.
        // (But skip this for perfectly specular BSDFs.)
        if (isect.bsdf->NumComponents(BxDFType(BSDF_ALL & ~BSDF_SPECULAR)) >
            0) {
            ++totalPaths;
            Spectrum Ld = beta * UniformSampleOneLight(isect, scene, arena,
                                                       sampler, false, distrib);
            VLOG(2) << "Sampled direct lighting Ld = " << Ld;
            if (Ld.IsBlack()) ++zeroRadiancePaths;
            CHECK_GE(Ld.y(), 0.f);
            L += Ld;
        }

        // Sample BSDF to get new path direction
        Vector3f wo = -ray.d, wi;
        Float pdf;
        BxDFType flags;
        Spectrum f = isect.bsdf->Sample_f(wo, &wi, sampler.Get2D(), &pdf,
                                          BSDF_ALL, &flags);
        VLOG(2) << "Sampled BSDF, f = " << f << ", pdf = " << pdf;
        if (f.IsBlack() || pdf == 0.f) break;
        beta *= f * AbsDot(wi, isect.shading.n) / pdf;
        VLOG(2) << "Updated beta = " << beta;
        CHECK_GE(beta.y(), 0.f);
        DCHECK(!std::isinf(beta.y()));
        specularBounce = (flags & BSDF_SPECULAR) != 0;
        if ((flags & BSDF_SPECULAR) && (flags & BSDF_TRANSMISSION)) {
            Float eta = isect.bsdf->eta;
            // Update the term that tracks radiance scaling for refraction
            // depending on whether the ray is entering or leaving the
            // medium.
            etaScale *= (Dot(wo, isect.n) > 0) ? (eta * eta) : 1 / (eta * eta);
        }
        ray = isect.SpawnRay(wi);

        // Account for subsurface scattering, if applicable
        if (isect.bssrdf && (flags & BSDF_TRANSMISSION)) {
            // Importance sample the BSSRDF
            SurfaceInteraction pi;
            Spectrum S = isect.bssrdf->Sample_S(
                scene, sampler.Get1D(), sampler.Get2D(), arena, &pi, &pdf);
            DCHECK(!std::isinf(beta.y()));
            if (S.IsBlack() || pdf == 0) break;
            beta *= S / pdf;

            // Account for the direct subsurface scattering component
            L += beta * UniformSampleOneLight(pi, scene, arena, sampler, false,
                                              lightDistribution->Lookup(pi.p));

            // Account for the indirect subsurface scattering component
            Spectrum f = pi.bsdf->Sample_f(pi.wo, &wi, sampler.Get2D(), &pdf,
                                           BSDF_ALL, &flags);
            if (f.IsBlack() || pdf == 0) break;
            beta *= f * AbsDot(wi, pi.shading.n) / pdf;
            DCHECK(!std::isinf(beta.y()));
            specularBounce = (flags & BSDF_SPECULAR) != 0;
            ray = pi.SpawnRay(wi);
        }

        // Possibly terminate the path with Russian roulette.
        // Factor out radiance scaling due to refraction in rrBeta.
        Spectrum rrBeta = beta * etaScale;
        if (rrBeta.MaxComponentValue() < rrThreshold && bounces > 3) {
            Float q = std::max((Float).05, 1 - rrBeta.MaxComponentValue());
            if (sampler.Get1D() < q) break;
            beta /= 1 - q;
            DCHECK(!std::isinf(beta.y()));
        }
    }
    ReportValue(pathLength, bounces);
    return L;
}

PathIntegrator *CreatePathIntegrator(const ParamSet &params,
                                     std::shared_ptr<Sampler> sampler,
                                     std::shared_ptr<const Camera> camera) {
    int maxDepth = params.FindOneInt("maxdepth", 5);
    int np;
    const int *pb = params.FindInt("pixelbounds", &np);
    Bounds2i pixelBounds = camera->film->GetSampleBounds();
    if (pb) {
        if (np != 4)
            Error("Expected four values for \"pixelbounds\" parameter. Got %d.",
                  np);
        else {
            pixelBounds = Intersect(pixelBounds,
                                    Bounds2i{{pb[0], pb[2]}, {pb[1], pb[3]}});
            if (pixelBounds.Area() == 0)
                Error("Degenerate \"pixelbounds\" specified.");
        }
    }
    Float rrThreshold = params.FindOneFloat("rrthreshold", 1.);
    std::string lightStrategy =
        params.FindOneString("lightsamplestrategy", "spatial");
    return new PathIntegrator(maxDepth, camera, sampler, pixelBounds,
                              rrThreshold, lightStrategy);
}

}  // namespace pbrt
