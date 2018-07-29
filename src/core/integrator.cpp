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
#include "hj.h"
#include "imageio.h"
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

std::vector<Spectrum> SamplerIntegrator::processPixel(
    Point2i pixel, uint32_t &remainingSampleNum, const Scene &scene,
    std::shared_ptr<Sampler> &tileSampler, MemoryArena &arena,
    std::unique_ptr<FilmTile> &filmTile) {
    std::vector<Spectrum> radianceValues(remainingSampleNum);

    for (uint32_t &sampleIndex = remainingSampleNum; sampleIndex > 0;
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
        filmTile->AddSample(cameraSample.pFilm, L, rayWeight);

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
    const std::string PATH_GT = "cornell_depth1_ss_gt.exr";

    Executor exec;

    // dummy params to remove cold cache
    exec.addParams({128, ASMethod::Rvariance, 0.0, 128 * 8, 2});

    exec.addParams({158, ASMethod::Efficiency, 0.0, 128 * 8, 2});
    exec.addParams({158, ASMethod::Efficiency, 25.0, 128 * 8, 2});
    exec.addParams({158, ASMethod::Efficiency, 10.0, 128 * 8, 2});
    exec.addParams({158, ASMethod::Efficiency, 5.0, 128 * 8, 2});
    exec.addParams({158, ASMethod::Efficiency, 1.0, 128 * 8, 2});

    /*exec.addParams({154, ASMethod::Efficiency, 0.0, 2});
    exec.addParams({128, ASMethod::Rvariance, 0.0, 2});

    exec.addParams({154, ASMethod::Efficiency, 10.0, 2});
    exec.addParams({128, ASMethod::Rvariance, 10.0, 2});

    exec.addParams({154, ASMethod::Efficiency, 5.0, 2});
    exec.addParams({128, ASMethod::Rvariance, 5.0, 2});

    exec.addParams({154, ASMethod::Efficiency, 1.0, 2});
    exec.addParams({128, ASMethod::Rvariance, 1.0, 2});*/

    for (int exe = 0; exe < exec.getNum(); ++exe) {
        ExecutionParams params = exec.getParams(exe);
        std::string path = "results\\" + params.getDirectoryName() + "\\";
        createDirectory(path);

        int SPP = params.spp;
        camera->film->filename = path + params.getDirectoryName() + ".exr";

        // Film should be clear before processing on new params
        camera->film->Clear();

        const int BATCH_SIZE = SPP / params.batch;
        if (SPP % BATCH_SIZE != 0) {
            printf("SPP(%d) is not dividible by BATCH_SIZE(%d)",
                   sampler->samplesPerPixel, BATCH_SIZE);
            exit(1);
        }
        const int BATCH_NUM = std::div(SPP, BATCH_SIZE).quot;

        // to exclude cold cache latency, remove some rows
        const int START_ROW = 2;
        const int OFFSET = START_ROW * imageX;
        const int SAMPLES_PER_BATCH =
            BATCH_SIZE * imageX * (imageY - START_ROW);

        // std::vector<std::vector<std::unique_ptr<FilmTile>>>
        // filmTileArray(imageY);
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

        std::vector<PixelEfficiency> efficiencyList;
        {
            // set maximum spp with arbitrary larget number
            auto basicSampler =
                std::unique_ptr<Sampler>(new RandomSampler(1000000));
            for (int y = 0; y < imageY; ++y) {
                for (int x = 0; x < imageX; ++x) {
                    std::shared_ptr<Sampler> s =
                        basicSampler->Clone(y * imageX + x);
                    s->StartPixel(Point2i(x, y));
                    PixelEfficiency pEff(Point2i(x, y), s);
                    efficiencyList.push_back(pEff);
                }
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
                    [&](int64_t iter) {
                        auto &pEff = efficiencyList[iter];
                        uint32_t pixelIndex =
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

                        clock_t localTime, localStart = clock();
                        auto radianceValues = processPixel(
                            pixel, remainingSamples[pixelIndex], scene,
                            tileSampler, arena, filmTileArray[pixelIndex]);
                        localTime = std::clock() - localStart;

#ifdef ADAPTIVE_SAMPLING
                        // mean and variance of this batch
                        Spectrum sMean = std::accumulate(radianceValues.begin(),
                                                         radianceValues.end(),
                                                         Spectrum(0.f)) /
                                         radianceValues.size();
                        Spectrum sVariance = getVariance(radianceValues, sMean);

                        Float fMean = (sMean[0] + sMean[1] + sMean[2]) / 3;
                        Float fVariance =
                            (sVariance[0] + sVariance[1] + sVariance[2]) / 3;

                        pEff.update(radianceValues.size(), fMean, fVariance,
                                    localTime);

                        if (params.clampThreshold != 0) {
                            if (pEff.variance > params.clampThreshold) {
                                pEff.variance = params.clampThreshold;
                            }
                        }

                        // global efficiency
                        Float relativeVariance =
                            (pEff.variance / pow(pEff.mean + 0.0001, 2.0));

                        // different metrics
                        switch (params.method) {
                        case ASMethod::Rvariance:
                            pEff.efficiency = relativeVariance;
                            break;
                        case ASMethod::Efficiency:
                            pEff.efficiency = relativeVariance /
                                              std::max(localTime, clock_t(1));
                            break;
                        }

#endif

                        // counter[pixelIndex] = std::clock() - start;
                    },
                    efficiencyList.size());

                overheadStart = clock();

#ifdef ADAPTIVE_SAMPLING
                // do not sort at last iteration
                if (batch == BATCH_NUM) {
                    break;
                }

                printf("[Batch%d]\n", batch);

                // sort by efficiency
                std::sort(
                    efficiencyList.begin() + OFFSET, efficiencyList.end(),
                    [](const PixelEfficiency &lhs, const PixelEfficiency &rhs) {
                        return lhs.efficiency > rhs.efficiency;
                    });

                Float effSum = std::accumulate(
                    efficiencyList.begin() + OFFSET, efficiencyList.end(), 0.0,
                    [](const Float &a, const PixelEfficiency &b) {
                        return a + b.efficiency;
                    });

                std::cout << "most efficient pixel: "
                          << efficiencyList[OFFSET].pixel << std::endl;
                printf("efficiency max(%f), sum(%f)\n",
                       efficiencyList[OFFSET].efficiency, effSum);
                printf("mean[%f], variance[%f]\n", efficiencyList[OFFSET].mean,
                       efficiencyList[OFFSET].variance);

                uint32_t sampleCounter = 0;
                for (size_t i = OFFSET; i < efficiencyList.size(); ++i) {
                    auto &pEff = efficiencyList[i];
                    int ind = pEff.pixel.y * imageX + pEff.pixel.x;
                    Float ratio = pEff.efficiency / effSum;
                    pEff.sampler->SetSampleNumber(0);
                    int candidate = std::floor(SAMPLES_PER_BATCH * ratio);

                    if (params.maxSPP > 0 && candidate > params.maxSPP) {
                        candidate = params.maxSPP;
                    }

                    remainingSamples[ind] = candidate;
                    sampleCounter += candidate;

                    totalSampleNum[ind] += candidate;
                }

                uint32_t leftovers = SAMPLES_PER_BATCH - sampleCounter;
                printf(
                    "samples_per_batch(%d), sampleCounter(%ld), "
                    "leftovers(%d)\n",
                    SAMPLES_PER_BATCH, sampleCounter, leftovers);

                int ind = OFFSET;
                for (uint32_t &i = leftovers; i > 0; --i, ++ind) {
                    if (ind >= efficiencyList.size()) ind = OFFSET;
                    remainingSamples[efficiencyList[ind].pixel.y * imageX +
                                     efficiencyList[ind].pixel.x]++;
                }

#else
                for (size_t i = OFFSET; i < remainingSamples.size(); ++i) {
                    auto &pEff = efficiencyList[i];
                    remainingSamples[pEff.pixel.y * imageX + pEff.pixel.x] =
                        BATCH_SIZE;
                }
#endif
                overheadTime += clock() - overheadStart;
            }
            // reporter.Done();
        }

#ifdef ADAPTIVE_SAMPLING
        // [Stats] ==================================================
        globalTime = std::clock() - globalStart - overheadTime;
        printf("------------[Statistics]------------\n");
        printf("Time for overhead: %fs\n",
               overheadTime / Float(CLOCKS_PER_SEC));
        printf("Time for loop: %fs\n\n", globalTime / Float(CLOCKS_PER_SEC));
        printf("Counted samples: %u\n",
               std::accumulate(globalSampleCounter.begin(),
                               globalSampleCounter.end(), 0));
        ExecutionResult result;
        result.time = globalTime / Float(CLOCKS_PER_SEC);
        exec.addResult(result);

        // [PRINT] ==================================================
        std::vector<Float> varianceMap(imageX * imageY);
        std::vector<Float> relVarianceMap(imageX * imageY);
        std::vector<Float> efficiencyMap(imageX * imageY);
        std::vector<Float> timeMap(imageX * imageY);

        for (auto pEff : efficiencyList) {
            varianceMap[pEff.pixel.y * imageX + pEff.pixel.x] = pEff.variance;
            relVarianceMap[pEff.pixel.y * imageX + pEff.pixel.x] =
                pEff.variance / pow(pEff.mean + 0.0001, 2.0);
            efficiencyMap[pEff.pixel.y * imageX + pEff.pixel.x] =
                pEff.efficiency;
            timeMap[pEff.pixel.y * imageX + pEff.pixel.x] = pEff.time;
        }

        auto timeIndicator =
            std::to_string(globalTime / Float(CLOCKS_PER_SEC)) + ".txt";
        writeText(path, timeIndicator.c_str(), std::vector<int>(), Point2i(),
                  0);
        /*writeText(path + "raynum.txt", totalSampleNum, Point2i(256, 256),
        OFFSET); writeText(path + "variance.txt", varianceMap, Point2i(256,
        256), OFFSET); writeText(path + "relVariance.txt", relVarianceMap,
        Point2i(256, 256), OFFSET); writeText(path + "efficiency.txt",
        efficiencyMap, Point2i(256, 256), OFFSET); writeText(path + "time.txt",
        timeMap, Point2i(256, 256), OFFSET);*/

        // [Create Image] =====================================
        writeImage(path, "raynum.exr", totalSampleNum, Point2i(256, 256),
                   OFFSET);
        writeImage(path, "variance.exr", varianceMap, Point2i(256, 256),
                   OFFSET);
        writeImage(path, "relVariance.exr", relVarianceMap, Point2i(256, 256),
                   OFFSET);
        writeImage(path, "efficiency.exr", efficiencyMap, Point2i(256, 256),
                   OFFSET);
        writeImage(path, "time.exr", timeMap, Point2i(256, 256), OFFSET);
#endif

        // Merge image tile into _Film_
        for (auto &filmTile : filmTileArray) {
            camera->film->MergeFilmTile(std::move(filmTile));
        }

        // Save final image after rendering
        camera->film->WriteImage();

        auto mse = diff2(PATH_GT, camera->film->filename);
        // printf("mse(%f), rmse(%f)\n", mse.first, mse.second);
        char tmp[255];
        sprintf(tmp, "mse(%.10f),rmse(%.9f).txt", mse.first, mse.second);
        writeText(path, tmp, std::vector<Float>(), Point2i(), 0);
    }

    std::cout << "Rendering finished" << std::endl;
    LOG(INFO) << "Rendering finished";
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
