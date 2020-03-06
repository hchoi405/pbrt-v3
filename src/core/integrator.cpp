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
#include "integrator.h"
#include <array>
#include <chrono>
#include <ctime>
#include <fstream>
#include <numeric>
#include "camera.h"
#include "film.h"
#include "hj.h"
#include "imageio.h"
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
using hclock = std::chrono::high_resolution_clock;
using duration = std::chrono::duration<float>;
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

#ifdef ADAPTIVE_SAMPLING
std::vector<Spectrum> SamplerIntegrator::processPixel(
    int batch, Point2i pixel, uint64_t &remainingSampleNum, const Scene &scene,
    std::shared_ptr<Sampler> &tileSampler, MemoryArena &arena,
    std::unique_ptr<FilmTile> &filmTile) {
    std::vector<Spectrum> radianceValues(remainingSampleNum);

    for (uint64_t &sampleIndex = remainingSampleNum; sampleIndex > 0;
         --sampleIndex) {
        // Initialize _CameraSample_ for current sample
        CameraSample cameraSample = tileSampler->GetCameraSample(pixel);

        // Generate camera ray for current sample
        RayDifferential ray;
        Float rayWeight = camera->GenerateRayDifferential(cameraSample, &ray);
        ray.ScaleDifferentials(1 /
                               std::sqrt((Float)tileSampler->samplesPerPixel));
        ++nCameraRays;

        // Evaluate radiance along camera ray
        Spectrum L(0.f);
        if (rayWeight > 0) L = Li(ray, scene, *tileSampler, arena);

        // Issue warning if unexpected radiance value
        // returned
        if (L.HasNaNs()) {
            LOG(ERROR) << StringPrintf(
                "Not-a-number radiance value returned "
                "for pixel (%d, %d), sample %d. Setting to "
                "black.",
                pixel.x, pixel.y, (int)tileSampler->CurrentSampleNumber());
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
                pixel.x, pixel.y, (int)tileSampler->CurrentSampleNumber());
            L = Spectrum(0.f);
        }
        VLOG(1) << "Camera sample: " << cameraSample << " -> ray: " << ray
                << " -> L = " << L;

        radianceValues[sampleIndex - 1] = L;

        // Add camera ray's contribution to image
        // Exclude the initial sampling becaust it is only used for stat
        // estimation
        if (batch > 1) filmTile->AddSample(cameraSample.pFilm, L, rayWeight);

        // Free _MemoryArena_ memory from computing image
        arena.Reset();

        // reporter.Update();

        if (!tileSampler->StartNextSample()) {
            if (sampleIndex != 1) {
                std::cout << pixel << "ERROR! Lack of samples\n";
            }
        }
    }

    return radianceValues;
}

// SamplerIntegrator Method Definitions
void SamplerIntegrator::Render(const Scene &scene) {
    Preprocess(scene, *sampler);
    std::ios::sync_with_stdio(false);

    // Compute number of tiles, _nTiles_, to use for parallel rendering
    Bounds2i sampleBounds = camera->film->GetSampleBounds();
    Vector2i sampleExtent = sampleBounds.Diagonal();
    int imageX = sampleBounds.pMax[0];
    int imageY = sampleBounds.pMax[1];
    const int tileSize = 1;
    Point2i nTiles((sampleExtent.x + tileSize - 1) / tileSize,
                   (sampleExtent.y + tileSize - 1) / tileSize);

    std::string filename = camera->film->filename;
    const std::string PATH_GT = "gt_ceramic_rgb.exr";

    Executor exec;

    // dummy params to remove cold cache
    exec.addParams({2, 3, ASMethod::Efficiency, 0.999, 0, 1});

    // {initialSpp, SPP, Metric, clampThreshold, maxSppRatio, batchNum}
    exec.addParams({16, 16, ASMethod::Rvariance, 0.99, 0, 1});
    exec.addParams({16, 8192, ASMethod::Rvariance, 0.99, 0, 1});
    exec.addParams({8192, 16, ASMethod::Rvariance, 0.99, 0, 1});
    exec.addParams({8192, 8192, ASMethod::Rvariance, 0.99, 0, 1});

    exec.addParams({16, 16, ASMethod::Efficiency, 0.99, 0, 1});
    exec.addParams({16, 8192, ASMethod::Efficiency, 0.99, 0, 1});
    exec.addParams({8192, 16, ASMethod::Efficiency, 0.99, 0, 1});
    exec.addParams({8192, 8192, ASMethod::Efficiency, 0.99, 0, 1});

    exec.addParams({16, 16, ASMethod::Time, 0.99, 0, 1});
    exec.addParams({16, 8192, ASMethod::Time, 0.99, 0, 1});
    exec.addParams({8192, 16, ASMethod::Time, 0.99, 0, 1});
    exec.addParams({8192, 8192, ASMethod::Time, 0.99, 0, 1});

    for (int exe = 0; exe < exec.getNum(); ++exe) {
        const ExecutionParams params = exec.getParams(exe);
        std::cout << "----------[ " << params.getDirectoryName()
                  << " ] ----------" << std::endl;
        std::string path = "results/" + params.getDirectoryName() + "/";
        createDirectory(path);

        uint64_t SPP = params.spp;
        camera->film->filename = path + params.getDirectoryName() + ".exr";

        // Film should be clear before processing on new params
        camera->film->Clear();

        // exclude initial sampling
        bool useInitialSampling = params.initialSpp > 0;

        const int BATCH_SIZE = SPP / params.batchNum;
        if (SPP % BATCH_SIZE != 0) {
            printf("SPP(%lu) is not dividible by BATCH_SIZE(%d)", SPP,
                   BATCH_SIZE);
            std::cout << std::endl;
            continue;
        }

        // include initial sampling
        const int BATCH_NUM =
            std::div(SPP, BATCH_SIZE).quot + (useInitialSampling ? 1 : 0);

        std::vector<uint64_t> batchSizes(BATCH_NUM, BATCH_SIZE);
        if (useInitialSampling) batchSizes[0] = params.initialSpp;

        std::vector<std::unique_ptr<FilmTile>> filmTileArray;
        for (int y = 0; y < imageY; ++y) {
            for (int x = 0; x < imageX; ++x) {
                // Compute sample bounds for pixel
                int x0 = sampleBounds.pMin.x + x * tileSize;
                int x1 = std::min(x0 + tileSize, sampleBounds.pMax.x);
                int y0 = sampleBounds.pMin.y + y * tileSize;
                int y1 = std::min(y0 + tileSize, sampleBounds.pMax.y);
                // size of tileBounds (1,1) for pixel-based loop
                Bounds2i tileBounds(Point2i(x0, y0), Point2i(x1, y1));
                filmTileArray.push_back(camera->film->GetFilmTile(tileBounds));
            }
        }

        std::vector<PixelEfficiency> pixelList;
        {
            // set maximum spp with arbitrary larget number
            auto basicSampler =
                std::unique_ptr<Sampler>(new RandomSampler(100000000));
            for (int y = 0; y < imageY; ++y) {
                for (int x = 0; x < imageX; ++x) {
                    std::shared_ptr<Sampler> s =
                        basicSampler->Clone(y * imageX + x);
                    s->StartPixel(Point2i(x, y));
                    PixelEfficiency pEff(Point2i(x, y), s);
                    pixelList.push_back(pEff);
                }
            }
        }

        std::vector<uint64_t> remainingSamples(imageX * imageY, batchSizes[0]);
        std::vector<uint64_t> totalSampleNum(imageX * imageY, 0);
        std::vector<Float> globalVariance(imageX * imageY, 0);
        std::vector<Float> globalTime(imageX * imageY, 0);

        auto overheadStart = hclock::now();
        duration overheadTime;
        auto renderingStart = hclock::now();
        auto globalStart = hclock::now();
        duration globalTimeCounter = (hclock::now() - globalStart);
        std::vector<uint64_t> globalSampleCounter(MaxThreadIndex());
        {
            // ProgressReporter reporter(imageX * imageY * SPP, "Rendering");
            for (int batch = 1; batch <= BATCH_NUM; ++batch) {
                printf("\t[Batch %d with %lu samples]\n", batch,
                       batchSizes[batch - 1]);
                // batch=1 : initial sampling
                // batch>1 : rendering
                renderingStart = hclock::now();
                ParallelFor(
                    [&](int64_t iter) {
                        auto &pEff = pixelList[iter];
                        uint64_t pixelIndex =
                            pEff.pixel.y * imageX + pEff.pixel.x;
                        Point2i pixel = pEff.pixel;

                        // do not proceed
                        if (remainingSamples[pixelIndex] == 0) return;

                        // count samples
                        globalSampleCounter[ThreadIndex] +=
                            remainingSamples[pixelIndex];

                        // Allocate _MemoryArena_ for pixel
                        MemoryArena arena;

                        // Get sampler instance for pixel
                        std::shared_ptr<Sampler> &tileSampler = pEff.sampler;

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

                        // clock_t localTime, localStart = clock();
                        Float localTime;
                        auto localStart = hclock::now();
                        auto radianceValues = processPixel(
                            batch, pixel, remainingSamples[pixelIndex], scene,
                            tileSampler, arena, filmTileArray[pixelIndex]);
                        localTime =
                            Float((hclock::now() - localStart).count()) /
                            1000;  // in micro seconds

                        // mean and variance of this batch
                        Spectrum sMean = std::accumulate(radianceValues.begin(),
                                                         radianceValues.end(),
                                                         Spectrum(0.f)) /
                                         radianceValues.size();
                        Float fMean = sMean.y();

                        std::vector<Float> luminanceValues(
                            radianceValues.size());
                        for (int i = 0; i < radianceValues.size(); ++i)
                            luminanceValues[i] = radianceValues[i].y();
                        Float fVariance = getVariance(luminanceValues, fMean);

                        globalVariance[pixelIndex] = fVariance;
                        globalTime[pixelIndex] = localTime;

                        // Do not update at last iteration
                        if (batch != BATCH_NUM) {
                            if (radianceValues.size() <= 1) {
                                std::cout << "Cannot estimate using less than "
                                             "1 sample"
                                          << std::endl;
                            }
                            pEff.updateStats(radianceValues.size(), fMean,
                                             fVariance, localTime);
                        }
                        // counter[pielIndex] = std::clock() - start;
                    },
                    pixelList.size());

                duration batchRenderingTime = hclock::now() - renderingStart;
                // Exclude the time of initial sampling (1st batch)
                if (batch > 1)
                    globalTimeCounter += batchRenderingTime;

                // do not sort at last iteration
                if (batch == BATCH_NUM) {
                    break;
                }

                overheadStart = hclock::now();

                // sort variance
                auto varIndice = orderedIndice(globalVariance);
                auto timeIndice = orderedIndice(globalTime);

                // clamp variance
                int clampingNum =
                    std::floor(pixelList.size() * (1 - params.clampThreshold));
                Float varThreshold = globalVariance[varIndice[clampingNum]];
                Float timeThreshold = globalTime[timeIndice[clampingNum]];

                // Update efficiency
                std::for_each(pixelList.begin(), pixelList.end(),
                              [&varThreshold, &timeThreshold,
                               &params](PixelEfficiency &pEff) {
                                  if (pEff.variance > varThreshold)
                                      pEff.variance = varThreshold;
                                  if (pEff.time > timeThreshold)
                                      pEff.time = timeThreshold;
                                  pEff.updateEfficiency(params.method);
                              });

                // sort by efficiency
                std::sort(
                    pixelList.begin(), pixelList.end(),
                    [](const PixelEfficiency &lhs, const PixelEfficiency &rhs) {
                        return lhs.efficiency > rhs.efficiency;
                    });

                Float effSum = std::accumulate(
                    pixelList.begin(), pixelList.end(), Float(0.0),
                    [](const Float &a, const PixelEfficiency &b) {
                        return a + b.efficiency;
                    });
                if (effSum == Float(0.0)) {
                    std::cout << "Error, sum of efficiency is zero"
                              << std::endl;
                    break;
                }

                // Normalize the efficiency
                for (auto &pixel : pixelList) {
                    pixel.efficiency /= effSum;
                }
      
                std::cout << "most efficient pixel: " << pixelList[0].pixel
                          << std::endl;
                printf("efficiency max(%f), sum(%f)\n", pixelList[0].efficiency,
                       effSum);
                printf("mean[%f], variance[%f]\n", pixelList[0].mean,
                       pixelList[0].variance);

                const uint64_t SAMPLES_PER_BATCH =
                    batchSizes[batch] * imageX * imageY;
                // Remaining samples after assigning 1 SPP for all pixels
                auto samplerPerBatchRemain =
                    SAMPLES_PER_BATCH - pixelList.size();

                // Calculate the SPP for next iteration
                uint64_t sampleCounter = 0;
                for (size_t i = 0; i < pixelList.size(); ++i) {
                    auto &pEff = pixelList[i];
                    int ind = pEff.pixel.y * imageX + pEff.pixel.x;
                    Float ratio = pEff.efficiency;
                    pEff.sampler->SetSampleNumber(0);
                    uint64_t candidate = std::max(
                        (uint64_t)std::floor(samplerPerBatchRemain * ratio),
                        uint64_t(1));

                    // spp clamping (not used)
                    // if (params.maxSppRatio > 0 &&
                    //     candidate > params.spp * params.maxSppRatio) {
                    //     candidate = params.spp * params.maxSppRatio;
                    // }

                    remainingSamples[ind] = candidate;
                    sampleCounter += candidate;
                    totalSampleNum[ind] += candidate;
                }

                int64_t leftovers = SAMPLES_PER_BATCH - sampleCounter;
                printf(
                    "samples_per_batch(%lu), sampleCounter(%lu), "
                    "leftovers(%ld)\n",
                    SAMPLES_PER_BATCH, sampleCounter, leftovers);
                if (leftovers < 0) {
                    std::cout << "Error, leftovers is negative: " << leftovers
                              << std::endl;
                    break;
                }
                // Distribute leftovers to pixels randomly
                for (int64_t &i = leftovers; i > 0; --i) {
                    int ind =
                        int((rand() / Float(RAND_MAX)) * pixelList.size());
                    remainingSamples[ind]++;
                }

                overheadTime += hclock::now() - overheadStart;
            }
            // reporter.Done();
        }


        // [Stats] ==================================================
        printf("\t[Statistics]\n");
        printf("Time without overhead %fs\n", globalTimeCounter.count());
        printf("Time for overhead: %fs\n", overheadTime.count());
        printf("Counted samples: %u\n",
               std::accumulate(globalSampleCounter.begin(),
                               globalSampleCounter.end(), 0));
        std::cout << std::endl;

        ExecutionResult result;
        result.time = globalTimeCounter.count();
        exec.addResult(result);

        // [PRINT] ==================================================
        std::vector<Float> varianceMap(imageX * imageY);
        std::vector<Float> relVarianceMap(imageX * imageY);
        std::vector<Float> efficiencyMap(imageX * imageY);
        std::vector<Float> timeMap(imageX * imageY);

        for (auto pEff : pixelList) {
            varianceMap[pEff.pixel.y * imageX + pEff.pixel.x] =
                pEff.variance / pEff.n;
            relVarianceMap[pEff.pixel.y * imageX + pEff.pixel.x] =
                pEff.variance / pEff.n / (pow(pEff.mean, 2.0) + +0.01);
            efficiencyMap[pEff.pixel.y * imageX + pEff.pixel.x] =
                pEff.efficiency;
            timeMap[pEff.pixel.y * imageX + pEff.pixel.x] = pEff.time / pEff.n;
        }

        auto timeIndicator = std::to_string(globalTimeCounter.count());
        auto tmpVec = std::vector<int>();
        writeText(path, timeIndicator.c_str(), tmpVec, Point2i(), 0);

        // [Create Text] =====================================
        // writeText(path, "raynum", totalSampleNum, Point2i(256, 256), 0);
        // writeText(path, "variance", varianceMap, Point2i(256, 256), 0);
        // writeText(path, "relVariance", relVarianceMap, Point2i(256, 256),
        //          0);
        // writeText(path, "efficiency", efficiencyMap, Point2i(256, 256),
        // 0);
        writeText(path, "time", timeMap, Point2i(imageX, imageY), 0);

        // [Create Image] =====================================
        writeImage(path, "raynum", totalSampleNum, Point2i(imageX, imageY), 0);
        writeImage(path, "variance", varianceMap, Point2i(imageX, imageY), 0);
        writeImage(path, "relVariance", relVarianceMap, Point2i(imageX, imageY),
                   0);
        writeImage(path, "efficiency", efficiencyMap, Point2i(imageX, imageY),
                   0);
        writeImage(path, "time", timeMap, Point2i(imageX, imageY), 0);

        // Merge image tile into _Film_
        for (auto &filmTile : filmTileArray) {
            camera->film->MergeFilmTile(std::move(filmTile));
        }

        // Save final image after rendering
        camera->film->WriteImage();

        auto mse = diff2(PATH_GT, camera->film->filename);
        // printf("mse(%f), rmse(%f)\n", mse.first, mse.second);
        char tmp[255];
        sprintf(tmp, "mse(%.10f),rmse(%.9f)", mse.first, mse.second);
        writeText(path, tmp, tmpVec, Point2i(), 0);
    }

    std::cout << "Rendering finished" << std::endl;
    LOG(INFO) << "Rendering finished";
}
#else
// SamplerIntegrator Method Definitions
void SamplerIntegrator::Render(const Scene &scene) {
    Preprocess(scene, *sampler);
    // Render image tiles in parallel

    // Compute number of tiles, _nTiles_, to use for parallel rendering
    Bounds2i sampleBounds = camera->film->GetSampleBounds();
    Vector2i sampleExtent = sampleBounds.Diagonal();
    const int tileSize = 16;
    Point2i nTiles((sampleExtent.x + tileSize - 1) / tileSize,
                   (sampleExtent.y + tileSize - 1) / tileSize);
    ProgressReporter reporter(nTiles.x * nTiles.y, "Rendering");
    {
        ParallelFor2D(
            [&](Point2i tile) {
                // Render section of image corresponding to _tile_

                // Allocate _MemoryArena_ for tile
                MemoryArena arena;

                // Get sampler instance for tile
                int seed = tile.y * nTiles.x + tile.x;
                std::unique_ptr<Sampler> tileSampler = sampler->Clone(seed);

                // Compute sample bounds for tile
                int x0 = sampleBounds.pMin.x + tile.x * tileSize;
                int x1 = std::min(x0 + tileSize, sampleBounds.pMax.x);
                int y0 = sampleBounds.pMin.y + tile.y * tileSize;
                int y1 = std::min(y0 + tileSize, sampleBounds.pMax.y);
                Bounds2i tileBounds(Point2i(x0, y0), Point2i(x1, y1));
                LOG(INFO) << "Starting image tile " << tileBounds;

                // Get _FilmTile_ for tile
                std::unique_ptr<FilmTile> filmTile =
                    camera->film->GetFilmTile(tileBounds);

                // Loop over pixels in tile to render them
                for (Point2i pixel : tileBounds) {
                    {
                        ProfilePhase pp(Prof::StartPixel);
                        tileSampler->StartPixel(pixel);
                    }

                    // Do this check after the StartPixel() call; this keeps
                    // the usage of RNG values from (most) Samplers that use
                    // RNGs consistent, which improves reproducability /
                    // debugging.
                    if (!InsideExclusive(pixel, pixelBounds)) continue;

                    do {
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

                        // Issue warning if unexpected radiance value returned
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

                        // Add camera ray's contribution to image
                        filmTile->AddSample(cameraSample.pFilm, L, rayWeight);

                        // Free _MemoryArena_ memory from computing image sample
                        // value
                        arena.Reset();
                    } while (tileSampler->StartNextSample());
                }
                LOG(INFO) << "Finished image tile " << tileBounds;

                // Merge image tile into _Film_
                camera->film->MergeFilmTile(std::move(filmTile));
                reporter.Update();
            },
            nTiles);
        reporter.Done();
    }
    LOG(INFO) << "Rendering finished";

    // Save final image after rendering
    camera->film->WriteImage();
}
#endif

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
    if (pdf > Float(0.0) && !f.IsBlack() && AbsDot(wi, ns) != Float(0.0)) {
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
    if (pdf > Float(0.0) && !f.IsBlack() && AbsDot(wi, ns) != Float(0.0)) {
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
              - pbrt computes the refracted ray as: \wi = -\eta \omega_o + [
              \eta (\wo \cdot \N) - \cos \theta_t ] \N It flips the normal to
              lie in the same hemisphere as \wo, and then \eta is the relative
              IOR from \wo's medium to \wi's medium.
              - If we denote the term in brackets by \mu, then we have: \wi =
              -\eta \omega_o + \mu \N
              - Now let's take the partial derivative. (We'll use "d" for
              \partial in the following for brevity.) We get: -\eta d\omega_o /
              dx + \mu dN/dx + d\mu/dx N.
              - We have the values of all of these except for d\mu/dx (using
              bits from the derivation of specularly reflected ray
              deifferentials).
              - The first term of d\mu/dx is easy: \eta d(\wo \cdot N)/dx. We
              already have d(\wo \cdot N)/dx.
              - The second term takes a little more work. We have:
                 \cos \theta_i = \sqrt{1 - \eta^2 (1 - (\wo \cdot N)^2)}.
                 Starting from (\wo \cdot N)^2 and reading outward, we have
              \cos^2 \theta_o, then \sin^2 \theta_o, then \sin^2 \theta_i (via
              Snell's law), then \cos^2 \theta_i and then \cos \theta_i.
              - Let's take the partial derivative of the sqrt expression. We
              get: 1 / 2 * 1 / \cos \theta_i * d/dx (1 - \eta^2 (1 - (\wo \cdot
              N)^2)).
              - That partial derivatve is equal to:
                d/dx \eta^2 (\wo \cdot N)^2 = 2 \eta^2 (\wo \cdot N) d/dx (\wo
              \cdot N).
              - Plugging it in, we have d\mu/dx =
                \eta d(\wo \cdot N)/dx - (\eta^2 (\wo \cdot N) d/dx (\wo \cdot
              N))/(-\wi \cdot N).
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
