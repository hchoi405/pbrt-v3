#define ADAPTIVE_SAMPLING

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

// core/integrator.cpp*
#include <array>
#include <ctime>
#include <fstream>
#include <numeric>
#include "camera.h"
#include "film.h"
#include "integrator.h"
#include "interaction.h"
#include "parallel.h"
#include "paramset.h"
#include "progressreporter.h"
#include "sampler.h"
#include "samplers/random.h"
#include "sampling.h"
#include "scene.h"
#include "stats.h"

namespace pbrt {

STAT_COUNTER("Integrator/Camera rays traced", nCameraRays);

// Integrator Method Definitions
Integrator::~Integrator() {}

// Integrator Utility Functions
Spectrum UniformSampleAllLights(const Interaction &it, const Scene &scene,
                                MemoryArena &arena, Sampler &sampler,
                                const std::vector<int> &nLightSamples,
                                bool handleMedia) {
    ProfilePhase p(Prof::DirectLighting);
    Spectrum L(0.f);
    for (size_t j = 0; j < scene.lights.size(); ++j) {
        // Accumulate contribution of _j_th light to _L_
        const std::shared_ptr<Light> &light = scene.lights[j];
        int nSamples = nLightSamples[j];
        const Point2f *uLightArray = sampler.Get2DArray(nSamples);
        const Point2f *uScatteringArray = sampler.Get2DArray(nSamples);
        if (!uLightArray || !uScatteringArray) {
            // Use a single sample for illumination from _light_
            Point2f uLight = sampler.Get2D();
            Point2f uScattering = sampler.Get2D();
            L += EstimateDirect(it, uScattering, *light, uLight, scene, sampler,
                                arena, handleMedia);
        } else {
            // Estimate direct lighting using sample arrays
            Spectrum Ld(0.f);
            for (int k = 0; k < nSamples; ++k)
                Ld += EstimateDirect(it, uScatteringArray[k], *light,
                                     uLightArray[k], scene, sampler, arena,
                                     handleMedia);
            L += Ld / nSamples;
        }
    }
    return L;
}

Spectrum UniformSampleOneLight(const Interaction &it, const Scene &scene,
                               MemoryArena &arena, Sampler &sampler,
                               bool handleMedia,
                               const Distribution1D *lightDistrib) {
    ProfilePhase p(Prof::DirectLighting);
    // Randomly choose a single light to sample, _light_
    int nLights = int(scene.lights.size());
    if (nLights == 0) return Spectrum(0.f);
    int lightNum;
    Float lightPdf;
    if (lightDistrib) {
        lightNum = lightDistrib->SampleDiscrete(sampler.Get1D(), &lightPdf);
        if (lightPdf == 0) return Spectrum(0.f);
    } else {
        lightNum = std::min((int)(sampler.Get1D() * nLights), nLights - 1);
        lightPdf = Float(1) / nLights;
    }
    const std::shared_ptr<Light> &light = scene.lights[lightNum];
    Point2f uLight = sampler.Get2D();
    Point2f uScattering = sampler.Get2D();
    return EstimateDirect(it, uScattering, *light, uLight, scene, sampler,
                          arena, handleMedia) /
           lightPdf;
}

Spectrum EstimateDirect(const Interaction &it, const Point2f &uScattering,
                        const Light &light, const Point2f &uLight,
                        const Scene &scene, Sampler &sampler,
                        MemoryArena &arena, bool handleMedia, bool specular) {
    BxDFType bsdfFlags =
        specular ? BSDF_ALL : BxDFType(BSDF_ALL & ~BSDF_SPECULAR);
    Spectrum Ld(0.f);
    // Sample light source with multiple importance sampling
    Vector3f wi;
    Float lightPdf = 0, scatteringPdf = 0;
    VisibilityTester visibility;
    Spectrum Li = light.Sample_Li(it, uLight, &wi, &lightPdf, &visibility);
    VLOG(2) << "EstimateDirect uLight:" << uLight << " -> Li: " << Li
            << ", wi: " << wi << ", pdf: " << lightPdf;
    if (lightPdf > 0 && !Li.IsBlack()) {
        // Compute BSDF or phase function's value for light sample
        Spectrum f;
        if (it.IsSurfaceInteraction()) {
            // Evaluate BSDF for light sampling strategy
            const SurfaceInteraction &isect = (const SurfaceInteraction &)it;
            f = isect.bsdf->f(isect.wo, wi, bsdfFlags) *
                AbsDot(wi, isect.shading.n);
            scatteringPdf = isect.bsdf->Pdf(isect.wo, wi, bsdfFlags);
            VLOG(2) << "  surf f*dot :" << f
                    << ", scatteringPdf: " << scatteringPdf;
        } else {
            // Evaluate phase function for light sampling strategy
            const MediumInteraction &mi = (const MediumInteraction &)it;
            Float p = mi.phase->p(mi.wo, wi);
            f = Spectrum(p);
            scatteringPdf = p;
            VLOG(2) << "  medium p: " << p;
        }
        if (!f.IsBlack()) {
            // Compute effect of visibility for light source sample
            if (handleMedia) {
                Li *= visibility.Tr(scene, sampler);
                VLOG(2) << "  after Tr, Li: " << Li;
            } else {
                if (!visibility.Unoccluded(scene)) {
                    VLOG(2) << "  shadow ray blocked";
                    Li = Spectrum(0.f);
                } else
                    VLOG(2) << "  shadow ray unoccluded";
            }

            // Add light's contribution to reflected radiance
            if (!Li.IsBlack()) {
                if (IsDeltaLight(light.flags))
                    Ld += f * Li / lightPdf;
                else {
                    Float weight =
                        PowerHeuristic(1, lightPdf, 1, scatteringPdf);
                    Ld += f * Li * weight / lightPdf;
                }
            }
        }
    }

    // Sample BSDF with multiple importance sampling
    if (!IsDeltaLight(light.flags)) {
        Spectrum f;
        bool sampledSpecular = false;
        if (it.IsSurfaceInteraction()) {
            // Sample scattered direction for surface interactions
            BxDFType sampledType;
            const SurfaceInteraction &isect = (const SurfaceInteraction &)it;
            f = isect.bsdf->Sample_f(isect.wo, &wi, uScattering, &scatteringPdf,
                                     bsdfFlags, &sampledType);
            f *= AbsDot(wi, isect.shading.n);
            sampledSpecular = (sampledType & BSDF_SPECULAR) != 0;
        } else {
            // Sample scattered direction for medium interactions
            const MediumInteraction &mi = (const MediumInteraction &)it;
            Float p = mi.phase->Sample_p(mi.wo, &wi, uScattering);
            f = Spectrum(p);
            scatteringPdf = p;
        }
        VLOG(2) << "  BSDF / phase sampling f: " << f
                << ", scatteringPdf: " << scatteringPdf;
        if (!f.IsBlack() && scatteringPdf > 0) {
            // Account for light contributions along sampled direction _wi_
            Float weight = 1;
            if (!sampledSpecular) {
                lightPdf = light.Pdf_Li(it, wi);
                if (lightPdf == 0) return Ld;
                weight = PowerHeuristic(1, scatteringPdf, 1, lightPdf);
            }

            // Find intersection and compute transmittance
            SurfaceInteraction lightIsect;
            Ray ray = it.SpawnRay(wi);
            Spectrum Tr(1.f);
            bool foundSurfaceInteraction =
                handleMedia ? scene.IntersectTr(ray, sampler, &lightIsect, &Tr)
                            : scene.Intersect(ray, &lightIsect);

            // Add light contribution from material sampling
            Spectrum Li(0.f);
            if (foundSurfaceInteraction) {
                if (lightIsect.primitive->GetAreaLight() == &light)
                    Li = lightIsect.Le(-wi);
            } else
                Li = light.Le(ray);
            if (!Li.IsBlack()) Ld += f * Li * Tr * weight / scatteringPdf;
        }
    }
    return Ld;
}

std::unique_ptr<Distribution1D> ComputeLightPowerDistribution(
    const Scene &scene) {
    if (scene.lights.empty()) return nullptr;
    std::vector<Float> lightPower;
    for (const auto &light : scene.lights)
        lightPower.push_back(light->Power().y());
    return std::unique_ptr<Distribution1D>(
        new Distribution1D(&lightPower[0], lightPower.size()));
}

template <typename T>
T getVariance(std::vector<T> &arr, T mean) {
    return std::accumulate(arr.begin(), arr.end(), T(0.f),
                           [&mean](const T &a, const T &b) {
                               // std::cout << a << ", " << b << std::endl;
                               return a + (b - mean) * (b - mean);
                           }) /
           arr.size();
}

void processPixel(Point2i pixel) {}

// SamplerIntegrator Method Definitions
void SamplerIntegrator::Render(const Scene &scene) {
    Preprocess(scene, *sampler);
    std::ios::sync_with_stdio(false);

    // Compute number of tiles, _nTiles_, to use for parallel rendering
    Bounds2i sampleBounds = camera->film->GetSampleBounds();
    Vector2i sampleExtent = sampleBounds.Diagonal();
    const int tileSize = 1;
    Point2i nTiles((sampleExtent.x + tileSize - 1) / tileSize,
                   (sampleExtent.y + tileSize - 1) / tileSize);

    const int SPP = sampler->samplesPerPixel;
    const int BATCH_SIZE = SPP / 2;  // SPP / 2
    if (SPP % BATCH_SIZE != 0) {
        printf("SPP(%d) is not dividible by BATCH_SIZE(%d)",
               sampler->samplesPerPixel, BATCH_SIZE);
        exit(1);
    }
    const int BATCH_NUM = std::div(SPP, BATCH_SIZE).quot;
    int imageX = sampleBounds.pMax[0];
    int imageY = sampleBounds.pMax[1];
    // to exclude cold cache latency, remove some rows
    const int START_ROW = 2;
    const int OFFSET = START_ROW * imageX;
    const int SAMPLES_PER_BATCH = BATCH_SIZE * imageX * (imageY - START_ROW);

    std::vector<Float> efficiencyCounter(imageX * imageY);
    std::vector<int> spps = {128, 256, 512, 1024, 2048, 4096};

    char filename[255];

    sprintf(filename, "raytime_spp%d.txt", sampler->samplesPerPixel);
    std::ofstream out(filename);

    /*std::vector<std::vector<std::unique_ptr<Sampler>>> pixelSamplerArray(
        imageY);*/
    std::vector<std::vector<std::unique_ptr<FilmTile>>> filmTileArray(imageY);
    for (int y = 0; y < imageY; ++y) {
        for (int x = 0; x < imageX; ++x) {
            // Compute sample bounds for pixel
            int x0 = sampleBounds.pMin.x + x * tileSize;
            int x1 = std::min(x0 + tileSize, sampleBounds.pMax.x);
            int y0 = sampleBounds.pMin.y + y * tileSize;
            int y1 = std::min(y0 + tileSize, sampleBounds.pMax.y);
            // size of tileBounds (1,1) for pixel-based loop
            Bounds2i tileBounds(Point2i(x0, y0), Point2i(x1, y1));
            filmTileArray[y].push_back(camera->film->GetFilmTile(tileBounds));
        }
    }

    struct PixelEfficiency {
        Point2i pixel;
        std::shared_ptr<Sampler> sampler;
        Float mean;
        Float variance;
        Float efficiency;
    };
    std::vector<PixelEfficiency> efficiencyMap;
    ParamSet param;
    // set maximum spp with arbitrary larget number
    int idata[1] = {1000000};
    param.AddInt("pixelsamples", std::unique_ptr<int[]>(idata), 1);
    auto basicSampler = CreateRandomSampler(param);
    for (int y = 0; y < imageY; ++y) {
        for (int x = 0; x < imageX; ++x) {
            auto s = basicSampler->Clone(y * imageX + x);
            s->StartPixel(Point2i(x, y));
            efficiencyMap.push_back({Point2i(x, y), std::move(s), Float(0.0),
                                     Float(0.0), Float(0.0)});
        }
    }

    std::vector<uint32_t> remainingSamples(imageX * imageY, BATCH_SIZE);
    std::vector<uint32_t> totalSampleNum(imageX * imageY, 0);

    clock_t globalTime = 0, globalStart = clock(), overheadTime = 0,
            overheadStart;
    std::vector<uint32_t> globalSampleCounter(MaxThreadIndex());
    {
        // ProgressReporter reporter(imageX * imageY * SPP, "Rendering");
        for (int batch = 1; batch <= BATCH_NUM; ++batch) {
            ParallelFor(
                [&](int64_t pIndex) {
                    Point2i pixel = efficiencyMap[pIndex].pixel;
                    uint32_t pixelIndex = pixel.y * imageX + pixel.x;

                    // do not proceed
                    if (remainingSamples[pixelIndex] == 0) return;

                    // Allocate _MemoryArena_ for pixel
                    MemoryArena arena;

                    // Get sampler instance for pixel
                    std::shared_ptr<Sampler> &tileSampler =
                        efficiencyMap[pixelIndex].sampler;

                    {
                        // ProfilePhase pp(Prof::StartPixel);
                        // tileSampler->StartPixel(pixel);
                    }

                    // Do this check after the StartPixel() call; this keeps
                    // the usage of RNG values from (most) Samplers that use
                    // RNGs consistent, which improves reproducability /
                    // debugging.
                    if (!InsideExclusive(pixel, pixelBounds)) {
                        return;
                    }

                    // count samples
                    globalSampleCounter[ThreadIndex] +=
                        remainingSamples[pixelIndex];

                    std::vector<Spectrum> radianceValues(
                        remainingSamples[pixelIndex]);
                    clock_t localTime, localStart = clock();

                    for (uint32_t &sampleIndex = remainingSamples[pixelIndex];
                         sampleIndex > 0; --sampleIndex) {
                        // Initialize _CameraSample_ for current sample
                        CameraSample cameraSample =
                            tileSampler->GetCameraSample(pixel);

                        // Generate camera ray for current sample
                        RayDifferential ray;
                        Float rayWeight =
                            camera->GenerateRayDifferential(cameraSample, &ray);
                        ray.ScaleDifferentials(
                            1 / std::sqrt((Float)tileSampler->samplesPerPixel));
                        ++nCameraRays;

                        // Evaluate radiance along camera ray
                        Spectrum L(0.f);
                        if (rayWeight > 0)
                            L = Li(ray, scene, *tileSampler, arena);

                        // Issue warning if unexpected radiance value
                        // returned
                        if (L.HasNaNs()) {
                            LOG(ERROR) << StringPrintf(
                                "Not-a-number radiance value returned "
                                "for pixel (%d, %d), sample %d. Setting to "
                                "black.",
                                pixel.x, pixel.y,
                                (int)tileSampler->CurrentSampleNumber());
                            L = Spectrum(0.f);
                        } else if (L.y() < -1e-5) {
                            LOG(ERROR) << StringPrintf(
                                "Negative luminance value, %f, returned "
                                "for pixel (%d, %d), sample %d. Setting to "
                                "black.",
                                L.y(), pixel.x, pixel.y,
                                (int)tileSampler->CurrentSampleNumber());
                            L = Spectrum(0.f);
                        } else if (std::isinf(L.y())) {
                            LOG(ERROR) << StringPrintf(
                                "Infinite luminance value returned "
                                "for pixel (%d, %d), sample %d. Setting to "
                                "black.",
                                pixel.x, pixel.y,
                                (int)tileSampler->CurrentSampleNumber());
                            L = Spectrum(0.f);
                        }
                        VLOG(1) << "Camera sample: " << cameraSample
                                << " -> ray: " << ray << " -> L = " << L;

                        radianceValues[sampleIndex - 1] = L;

                        // Add camera ray's contribution to image
                        filmTileArray[pixel.y][pixel.x]->AddSample(
                            cameraSample.pFilm, L, rayWeight);

                        // Free _MemoryArena_ memory from computing image
                        arena.Reset();

                        // reporter.Update();

                        if (!tileSampler->StartNextSample()) {
                            if (sampleIndex != 1) {
                                std::cout << pixel
                                          << "ERROR! Lack of samples\n";
                            }
                        }
                    }
                    localTime = std::clock() - localStart;

#ifdef ADAPTIVE_SAMPLING
                    // mean and variance of this batch
                    Spectrum sMean =
                        std::accumulate(radianceValues.begin(),
                                        radianceValues.end(), Spectrum(0.f)) /
                        radianceValues.size();
                    Spectrum sVariance = getVariance(radianceValues, sMean);
                    Float fMean = (sMean[0] + sMean[1] + sMean[2]) / 3;
                    Float fVariance =
                        (sVariance[0] + sVariance[1] + sVariance[2]) / 3;

                    // temporal variables for calculation
                    Float n = totalSampleNum[pixelIndex];
                    Float nVariance = efficiencyMap[pixelIndex].variance;
                    Float nMean = efficiencyMap[pixelIndex].mean;
                    Float m = radianceValues.size();
                    Float mVariance = fVariance;
                    Float mMean = fMean;

                    Float mnMean = (n * nMean + m * mMean) / (m + n);
                    Float mnVariance = (n * (nVariance + nMean * nMean) +
                                        m * (mVariance + mMean * mMean)) /
                                           (m + n) -
                                       mnMean * mnMean;

                    // update global mean and variance
                    efficiencyMap[pixelIndex].mean = mnMean;
                    efficiencyMap[pixelIndex].variance = mnVariance;

                    // global efficiency
                    Float relativeVariance =
                        (efficiencyMap[pixelIndex].variance /
                         (efficiencyMap[pixelIndex].mean + 0.0001));
                    efficiencyMap[pixelIndex].efficiency =
                        relativeVariance / std::max(localTime, clock_t(1));

                    // thresholding
                    /*if (efficiencyMap[pixelIndex].efficiency > 5) {
                        efficiencyMap[pixelIndex].efficiency = 5;
                    }*/
#endif

                    // counter[pixelIndex] = std::clock() - start;
                },
                efficiencyMap.size());

            overheadStart = clock();

#ifdef ADAPTIVE_SAMPLING
            // do not sort at last iteration
            if (batch == BATCH_NUM) {
                break;
            }
            
			printf("[Batch%d]\n", batch);

            // sort by efficiency
            std::sort(
                efficiencyMap.begin() + OFFSET, efficiencyMap.end(),
                [](const PixelEfficiency &lhs, const PixelEfficiency &rhs) {
                    return lhs.efficiency > rhs.efficiency;
                });

            /*for (size_t i = OFFSET; i < efficiencyMap.size(); ++i) {
                std::cout << efficiencyMap[i].efficiency << std::endl;
            }*/

            Float effSum = std::accumulate(
                efficiencyMap.begin() + OFFSET, efficiencyMap.end(), 0.0,
                [](const Float &a, const PixelEfficiency &b) {
                    return a + b.efficiency;
                });

            std::cout << "most efficient pixel: " << efficiencyMap[OFFSET].pixel
                      << std::endl;
            printf("efficiency max(%f), sum(%f)\n",
                   efficiencyMap[OFFSET].efficiency, effSum);
            printf("mean[%f], variance[%f]\n", efficiencyMap[OFFSET].mean,
                   efficiencyMap[OFFSET].variance);

            int zeroEffCounter = 0;
            for (size_t i = OFFSET; i < efficiencyMap.size(); ++i) {
                auto &pEff = efficiencyMap[i];
                Float ratio = pEff.efficiency / effSum;
                pEff.sampler->SetSampleNumber(0);
                remainingSamples[pEff.pixel.y * imageX + pEff.pixel.x] =
                    std::round(SAMPLES_PER_BATCH * ratio);
                if (remainingSamples[pEff.pixel.y * imageX + pEff.pixel.x] == 0)
                    zeroEffCounter++;

                totalSampleNum[pEff.pixel.y * imageX + pEff.pixel.x] +=
                    remainingSamples[pEff.pixel.y * imageX + pEff.pixel.x];
                efficiencyCounter[pEff.pixel.y * imageX + pEff.pixel.x] =
                    remainingSamples[pEff.pixel.y * imageX + pEff.pixel.x];
                // pEff.efficiency;
            }

            printf("number of zero efficiency: %d\n", zeroEffCounter);

#else
            for (size_t i = OFFSET; i < remainingSamples.size(); ++i) {
                auto &pEff = efficiencyMap[i];
                remainingSamples[pEff.pixel.y * imageX + pEff.pixel.x] =
                    BATCH_SIZE;
            }
#endif
            overheadTime += clock() - overheadStart;
        }
        // reporter.Done();
    }

    globalTime = std::clock() - globalStart - overheadTime;
    printf("------------[Statistics]------------\n");
    printf("Time for overhead: %fs\n", overheadTime / Float(CLOCKS_PER_SEC));
    printf("Time for loop: %fs\n\n", globalTime / Float(CLOCKS_PER_SEC));
    printf("Counted samples: %u\n",
           std::accumulate(globalSampleCounter.begin(),
                           globalSampleCounter.end(), 0));
    // Merge image tile into _Film_
    for (auto &arr : filmTileArray) {
        for (auto &filmTile : arr) {
            camera->film->MergeFilmTile(std::move(filmTile));
        }
    }

    for (int i = OFFSET; i < efficiencyCounter.size(); ++i) {
        out << totalSampleNum[i] << std::endl;
    }
    out.close();

    std::cout << "Rendering finished" << std::endl;
    LOG(INFO) << "Rendering finished";

    // Save final image after rendering
    camera->film->WriteImage();
}

Spectrum SamplerIntegrator::SpecularReflect(
    const RayDifferential &ray, const SurfaceInteraction &isect,
    const Scene &scene, Sampler &sampler, MemoryArena &arena, int depth) const {
    // Compute specular reflection direction _wi_ and BSDF value
    Vector3f wo = isect.wo, wi;
    Float pdf;
    BxDFType type = BxDFType(BSDF_REFLECTION | BSDF_SPECULAR);
    Spectrum f = isect.bsdf->Sample_f(wo, &wi, sampler.Get2D(), &pdf, type);

    // Return contribution of specular reflection
    const Normal3f &ns = isect.shading.n;
    if (pdf > 0.f && !f.IsBlack() && AbsDot(wi, ns) != 0.f) {
        // Compute ray differential _rd_ for specular reflection
        RayDifferential rd = isect.SpawnRay(wi);
        if (ray.hasDifferentials) {
            rd.hasDifferentials = true;
            rd.rxOrigin = isect.p + isect.dpdx;
            rd.ryOrigin = isect.p + isect.dpdy;
            // Compute differential reflected directions
            Normal3f dndx = isect.shading.dndu * isect.dudx +
                            isect.shading.dndv * isect.dvdx;
            Normal3f dndy = isect.shading.dndu * isect.dudy +
                            isect.shading.dndv * isect.dvdy;
            Vector3f dwodx = -ray.rxDirection - wo,
                     dwody = -ray.ryDirection - wo;
            Float dDNdx = Dot(dwodx, ns) + Dot(wo, dndx);
            Float dDNdy = Dot(dwody, ns) + Dot(wo, dndy);
            rd.rxDirection =
                wi - dwodx + 2.f * Vector3f(Dot(wo, ns) * dndx + dDNdx * ns);
            rd.ryDirection =
                wi - dwody + 2.f * Vector3f(Dot(wo, ns) * dndy + dDNdy * ns);
        }
        return f * Li(rd, scene, sampler, arena, depth + 1) * AbsDot(wi, ns) /
               pdf;
    } else
        return Spectrum(0.f);
}

Spectrum SamplerIntegrator::SpecularTransmit(
    const RayDifferential &ray, const SurfaceInteraction &isect,
    const Scene &scene, Sampler &sampler, MemoryArena &arena, int depth) const {
    Vector3f wo = isect.wo, wi;
    Float pdf;
    const Point3f &p = isect.p;
    const BSDF &bsdf = *isect.bsdf;
    Spectrum f = bsdf.Sample_f(wo, &wi, sampler.Get2D(), &pdf,
                               BxDFType(BSDF_TRANSMISSION | BSDF_SPECULAR));
    Spectrum L = Spectrum(0.f);
    Normal3f ns = isect.shading.n;
    if (pdf > 0.f && !f.IsBlack() && AbsDot(wi, ns) != 0.f) {
        // Compute ray differential _rd_ for specular transmission
        RayDifferential rd = isect.SpawnRay(wi);
        if (ray.hasDifferentials) {
            rd.hasDifferentials = true;
            rd.rxOrigin = p + isect.dpdx;
            rd.ryOrigin = p + isect.dpdy;

            Normal3f dndx = isect.shading.dndu * isect.dudx +
                            isect.shading.dndv * isect.dvdx;
            Normal3f dndy = isect.shading.dndu * isect.dudy +
                            isect.shading.dndv * isect.dvdy;

            // The BSDF stores the IOR of the interior of the object being
            // intersected.  Compute the relative IOR by first out by
            // assuming that the ray is entering the object.
            Float eta = 1 / bsdf.eta;
            if (Dot(wo, ns) < 0) {
                // If the ray isn't entering, then we need to invert the
                // relative IOR and negate the normal and its derivatives.
                eta = 1 / eta;
                ns = -ns;
                dndx = -dndx;
                dndy = -dndy;
            }

            /*
              Notes on the derivation:
              - pbrt computes the refracted ray as: \wi = -\eta \omega_o + [ \eta (\wo \cdot \N) - \cos \theta_t ] \N
                It flips the normal to lie in the same hemisphere as \wo, and then \eta is the relative IOR from
                \wo's medium to \wi's medium.
              - If we denote the term in brackets by \mu, then we have: \wi = -\eta \omega_o + \mu \N
              - Now let's take the partial derivative. (We'll use "d" for \partial in the following for brevity.)
                We get: -\eta d\omega_o / dx + \mu dN/dx + d\mu/dx N.
              - We have the values of all of these except for d\mu/dx (using bits from the derivation of specularly
                reflected ray deifferentials).
              - The first term of d\mu/dx is easy: \eta d(\wo \cdot N)/dx. We already have d(\wo \cdot N)/dx.
              - The second term takes a little more work. We have:
                 \cos \theta_i = \sqrt{1 - \eta^2 (1 - (\wo \cdot N)^2)}.
                 Starting from (\wo \cdot N)^2 and reading outward, we have \cos^2 \theta_o, then \sin^2 \theta_o,
                 then \sin^2 \theta_i (via Snell's law), then \cos^2 \theta_i and then \cos \theta_i.
              - Let's take the partial derivative of the sqrt expression. We get:
                1 / 2 * 1 / \cos \theta_i * d/dx (1 - \eta^2 (1 - (\wo \cdot N)^2)).
              - That partial derivatve is equal to:
                d/dx \eta^2 (\wo \cdot N)^2 = 2 \eta^2 (\wo \cdot N) d/dx (\wo \cdot N).
              - Plugging it in, we have d\mu/dx =
                \eta d(\wo \cdot N)/dx - (\eta^2 (\wo \cdot N) d/dx (\wo \cdot N))/(-\wi \cdot N).
             */
            Vector3f dwodx = -ray.rxDirection - wo,
                     dwody = -ray.ryDirection - wo;
            Float dDNdx = Dot(dwodx, ns) + Dot(wo, dndx);
            Float dDNdy = Dot(dwody, ns) + Dot(wo, dndy);

            Float mu = eta * Dot(wo, ns) - AbsDot(wi, ns);
            Float dmudx =
                (eta - (eta * eta * Dot(wo, ns)) / AbsDot(wi, ns)) * dDNdx;
            Float dmudy =
                (eta - (eta * eta * Dot(wo, ns)) / AbsDot(wi, ns)) * dDNdy;

            rd.rxDirection =
                wi - eta * dwodx + Vector3f(mu * dndx + dmudx * ns);
            rd.ryDirection =
                wi - eta * dwody + Vector3f(mu * dndy + dmudy * ns);
        }
        L = f * Li(rd, scene, sampler, arena, depth + 1) * AbsDot(wi, ns) / pdf;
    }
    return L;
}

}  // namespace pbrt
