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
#include <ctime>
#include <fstream>
#include <numeric>
#include "camera.h"
#include "film.h"
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

#include "nanoflann.hpp"
#include "utils.h"

#include <Windows.h>

namespace pbrt {
extern int counter;
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

struct PixelEfficiency {
    Point2i pixel;
    std::shared_ptr<Sampler> sampler;
    uint32_t n;
    Float mean;
    Float variance;
    Float time;
    Float efficiency;

    PixelEfficiency(Point2i _pixel, std::shared_ptr<Sampler> _sampler)
        : pixel(_pixel),
          sampler(_sampler),
          n(0),
          mean(Float(0)),
          variance(Float(0)),
          time(Float(0)),
          efficiency(Float(0)) {}

    void updateStats(size_t m, Float mMean, Float mVariance, Float mTime) {
        Float mn = n + m;
        Float mnMean = (n * mean + m * mMean) / mn;
        Float mnVariance =
            (n * (variance + mean * mean) + m * (mVariance + mMean * mMean)) /
                mn -
            mnMean * mnMean;
        Float mnTime = (n * time + m * mTime) / mn;

        this->n = mn;
        this->mean = mnMean;
        this->variance = mnVariance;
        this->time = mnTime;
    }

    void updateEfficiency(ASMethod method) {
        Float relativeVariance = variance / pow(mean + 0.0001, 2.0);

        // different metrics
        switch (method) {
        case ASMethod::Rvariance:
            efficiency = relativeVariance;
            break;
        case ASMethod::Efficiency:
            efficiency = relativeVariance / std::max(this->time, 1.0);
            break;
        }
    }
};

struct ExecutionResult {
    Float time;
};

struct ExecutionParams {
    int spp;
    ASMethod method;
    Float clampThreshold;
    int maxSppRatio;
    int batch;

    ExecutionResult result;

    std::string getDirectoryName() const {
        char tmp[255];
        std::string methodName;
        switch (method) {
        case ASMethod::Rvariance:
            methodName = std::string("rvar");
            break;
        case ASMethod::Efficiency:
            methodName = std::string("eff");
            break;
        }
        sprintf(tmp, "spp%d_%s_clamp%.4f_max%d", spp, methodName.c_str(),
                clampThreshold, maxSppRatio);
        return std::string(tmp);
    }
};

class Executor {
    std::vector<ExecutionParams> _params;
    std::vector<ExecutionResult> _results;

  public:
    Executor() { createFolder("results\\"); }
    size_t getNum() const { return _params.size(); }

    ExecutionParams getParams(int i) { return _params[i]; }

    void addParams(ExecutionParams params) { _params.push_back(params); }

    void addResult(ExecutionResult result) { _results.push_back(result); }

    static void createFolder(std::string path) {
        if (CreateDirectory(path.c_str(), NULL) ||
            ERROR_ALREADY_EXISTS == GetLastError()) {
        } else {
            std::cout << "failed to create directory: " << path << std::endl;
        }
    }
};

template <typename T>
T getVariance(std::vector<T> &arr, T mean) {
    return std::accumulate(arr.begin(), arr.end(), T(0.f),
                           [&mean](const T &a, const T &b) {
                               return a + (b - mean) * (b - mean);
                           }) /
           arr.size();
}

template <typename T>
void writeImage(std::string path, std::string filename, std::vector<T> &values,
                Point2i res, const int OFFSET) {
    writeImage(path.c_str(), filename.c_str(), values, res, OFFSET);
}
template <typename T>
void writeImage(const char path[], const char filename[],
                std::vector<T> &values, Point2i res, const int OFFSET) {
    std::unique_ptr<Float[]> rgb(new Float[3 * res.x * res.y]);

    auto minmax = std::minmax_element(values.begin(), values.end());
    Float maxValue = (minmax.second != values.end()) ? *minmax.second : 0.0;
    Float minValue = (minmax.first != values.end()) ? *minmax.first : 0.0;

    if (values.size() == res.y * res.x)
        for (int i = 0; i < res.y; ++i) {
            for (int j = 0; j < res.x; ++j) {
                int ind = i * res.x + j;
                rgb[3 * ind + 0] = Float(values[OFFSET + ind]);
                rgb[3 * ind + 1] = Float(values[OFFSET + ind]);
                rgb[3 * ind + 2] = Float(values[OFFSET + ind]);
            }
        }

    char newfilename[255];
    sprintf(newfilename, "%sstat_%s_[%.4f,%.4f].exr", path, filename, minValue,
            maxValue);
    WriteImage(newfilename, &rgb[0],
               Bounds2i(Point2i(0, 0), Point2i(res.x, res.y)),
               Point2i(res.x, res.y));
}

template <typename T>
void writeText(std::string path, std::string filename, std::vector<T> &values,
               Point2i res, const int OFFSET) {
    writeText(path.c_str(), filename.c_str(), values, res, OFFSET);
}

template <typename T>
void writeText(const char path[], const char filename[], std::vector<T> &values,
               Point2i res, const int OFFSET) {
    char newfilename[255];

    if (!values.empty()) {
        auto minmax = std::minmax_element(values.begin(), values.end());
        Float maxValue = *minmax.second;
        Float minValue = *minmax.first;
    }

    sprintf(newfilename, "%sstat_%s.txt", path, filename);
    std::ofstream out(newfilename);
    char tmp[255];
    for (int i = OFFSET; i < values.size(); ++i) {
        sprintf(tmp, "%f\n", Float(values[i]));
        out << tmp;
    }
    out.close();
}

std::pair<Float, Float> diff2(std::string gtPath, std::string in) {
    std::pair<Float, Float> result;

    float tol = 0.;
    const char *outfile = nullptr;

    const char *filename[2] = {gtPath.c_str(), in.c_str()};
    Point2i res[2];
    std::unique_ptr<RGBSpectrum[]> imgs[2] = {ReadImage(filename[0], &res[0]),
                                              ReadImage(filename[1], &res[1])};
    if (!imgs[0]) {
        fprintf(stderr, "%s: unable to read image\n", filename[0]);
        return result;
    }
    if (!imgs[1]) {
        fprintf(stderr, "%s: unable to read image\n", filename[1]);
        return result;
    }
    if (res[0] != res[1]) {
        fprintf(stderr,
                "imgtool: image resolutions don't match \"%s\": (%d, %d) "
                "\"%s\": (%d, %d)\n",
                filename[0], res[0].x, res[0].y, filename[1], res[1].x,
                res[1].y);
        return result;
    }

    double sum[2] = {0., 0.};
    int smallDiff = 0, bigDiff = 0;
    double rmse = 0.0;
    double mse = 0.0;
    for (int i = 0; i < res[0].x * res[0].y; ++i) {
        Float rgb[2][3];
        imgs[0][i].ToRGB(rgb[0]);
        imgs[1][i].ToRGB(rgb[1]);

        Float diffRGB[3];
        for (int c = 0; c < 3; ++c) {
            Float c0 = rgb[0][c], c1 = rgb[1][c];
            diffRGB[c] = std::abs(c0 - c1);

            if (c0 == 0 && c1 == 0) continue;

            sum[0] += c0;
            sum[1] += c1;

            float d = std::abs(c0 - c1) / c0;
            mse += (c0 - c1) * (c0 - c1);
            rmse += (c0 - c1) * (c0 - c1) / pow(c0 + 0.0001, 2.0);
            if (d > .005) ++smallDiff;
            if (d > .05) ++bigDiff;
        }
    }

    double avg[2] = {sum[0] / (3. * res[0].x * res[0].y),
                     sum[1] / (3. * res[0].x * res[0].y)};
    double avgDelta = (avg[0] - avg[1]) / std::min(avg[0], avg[1]);
    if ((tol == 0. && (bigDiff > 0 || smallDiff > 0)) ||
        (tol > 0. && 100.f * std::abs(avgDelta) > tol)) {
        result.first = mse / (3. * res[0].x * res[0].y);
        result.second = rmse / (3. * res[0].x * res[0].y);
        return result;
    }

    return result;
}

template <typename T>
std::vector<size_t> orderedIndice(std::vector<T> const &values) {
    std::vector<size_t> indices(values.size());
    std::iota(begin(indices), end(indices), static_cast<size_t>(0));

    std::sort(begin(indices), end(indices),
              [&](size_t a, size_t b) { return values[a] > values[b]; });
    return indices;
}

typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<Float, PointCloud<Float>>, PointCloud<Float>,
    3 /* dim */
    >
    my_kd_tree_t;
std::unique_ptr<my_kd_tree_t> kdtree;

std::vector<Spectrum> SamplerIntegrator::processPixel(
    Point2i pixel, uint32_t &remainingSampleNum, const Scene &scene,
    std::shared_ptr<Sampler> &tileSampler, MemoryArena &arena,
    std::unique_ptr<FilmTile> &filmTile, std::vector<PointInfo> &pointInfoList,
    int batch) {
    // vector for radiance values
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

        // std::cout << "batch: " << batch << std::endl;
        // Evaluate radiance along camera ray
        Spectrum L(0.f);
        if (rayWeight > 0)
            L = Li2(ray, scene, *tileSampler, arena, pointInfoList, batch);

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

    int rvarSpp = 128;
    int effSpp = 128;
    // dummy params to remove cold cache
    exec.addParams({rvarSpp, ASMethod::Rvariance, 1.0, 0, 2});

    exec.addParams({rvarSpp, ASMethod::Rvariance, 1.0, 0, 2});
    exec.addParams({effSpp, ASMethod::Efficiency, 1.0, 0, 2});
    // exec.addParams({rvarSpp, ASMethod::Rvariance, 0.9999, 0, 2});
    // exec.addParams({effSpp, ASMethod::Efficiency, 0.9999, 0, 2});
    // exec.addParams({rvarSpp, ASMethod::Rvariance, 0.999, 0, 2});
    // exec.addParams({effSpp, ASMethod::Efficiency, 0.999, 0, 2});
    // exec.addParams({rvarSpp, ASMethod::Rvariance, 0.99, 0, 2});
    // exec.addParams({effSpp, ASMethod::Efficiency, 0.99, 0, 2});

    for (int exe = 0; exe < exec.getNum(); ++exe) {
        ExecutionParams params = exec.getParams(exe);
        currentMethod = params.method;

        // initialize global variables
        counter = 0;
        delete kdtree.release();

        // create folder for this params
        std::string path = "results\\" + params.getDirectoryName() + "\\";
        Executor::createFolder(path);

        int SPP = params.spp;
        camera->film->filename = path + params.getDirectoryName() + ".exr";

        // Film should be clear before processing on new params
        camera->film->Clear();

        if (SPP % params.batch != 0) {
            printf("SPP(%d) is not dividible by BATCH_SIZE(%d)", SPP,
                   params.batch);
            exit(1);
        }
        const int BATCH_SIZE = SPP / params.batch;

        const int SAMPLES_PER_BATCH = BATCH_SIZE * imageX * imageY;

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
        std::vector<Float> globalVariance(imageX * imageY, 0);

        clock_t globalTime = 0, globalStart = clock();
        clock_t overheadTime = 0;

        std::vector<uint32_t> globalSampleCounter(MaxThreadIndex());
        std::vector<std::vector<PointInfo>> pointInfoList(MaxThreadIndex());
        std::vector<PointInfo> globalPointInfoList;
        PointCloud<Float> cloud;
        {
            bool estimate = true;
            // ProgressReporter reporter(imageX * imageY * SPP, "Rendering");
            for (int batch = 1; batch <= params.batch; ++batch) {
                printf("--------------- [Batch%d] ---------------\n", batch);
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

                        // ray tracing and shading
                        std::vector<PointInfo> *currentPointInfoList = nullptr;
                        if (estimate) {
                            currentPointInfoList = &pointInfoList[ThreadIndex];
                        } else {
                            currentPointInfoList = &globalPointInfoList;
                        }
                        auto radianceValues = processPixel(
                            pixel, remainingSamples[pixelIndex], scene,
                            tileSampler, arena, filmTileArray[pixelIndex],
                            *currentPointInfoList, batch);

                        localTime = std::clock() - localStart;

                        // mean and variance of this batch
                        Spectrum sMean = std::accumulate(radianceValues.begin(),
                                                         radianceValues.end(),
                                                         Spectrum(0.f)) /
                                         radianceValues.size();
                        Spectrum sVariance = getVariance(radianceValues, sMean);

                        Float fMean = (sMean[0] + sMean[1] + sMean[2]) / 3;
                        Float fVariance =
                            (sVariance[0] + sVariance[1] + sVariance[2]) / 3;
                        globalVariance[pixelIndex] = fVariance;

                        if (batch != params.batch) {
                            pEff.updateStats(radianceValues.size(), fMean,
                                             fVariance, localTime);
                        }
                        // counter[pixelIndex] = std::clock() - start;
                    },
                    efficiencyList.size());

                // Do estimation and reconstruction alternately
                estimate = !estimate;

                std::cout << "zero counter: " << counter << std::endl;

                // do not make sampling map at last iteration
                if (batch < params.batch) {
                    printf("Start Adaptive Sampling\n");
                    clock_t overheadStart = clock();

#ifdef IMAGE_SPACE
                    // get variance threshold
                    auto indice = orderedIndice(globalVariance);
                    int clampingNum = std::floor(efficiencyList.size() *
                                                 (1 - params.clampThreshold));
                    Float threshold = globalVariance[indice[clampingNum]];

                    // Update efficiency
                    std::for_each(efficiencyList.begin(), efficiencyList.end(),
                                  [&threshold, &params](PixelEfficiency &pEff) {
                                      if (pEff.variance > threshold)
                                          pEff.variance = threshold;
                                      pEff.updateEfficiency(params.method);
                                  });

                    // sort by efficiency
                    std::sort(efficiencyList.begin(), efficiencyList.end(),
                              [](const PixelEfficiency &lhs,
                                 const PixelEfficiency &rhs) {
                                  return lhs.efficiency > rhs.efficiency;
                              });
#else

                    // push points to k-d tree
                    globalPointInfoList.clear();
                    for (int i = 0; i < MaxThreadIndex(); ++i) {
                        globalPointInfoList.insert(globalPointInfoList.end(),
                                                   pointInfoList[i].begin(),
                                                   pointInfoList[i].end());
                        for (auto pointInfo : pointInfoList[i]) {
                            auto &p = pointInfo.point;
                            cloud.pts.push_back({p.x, p.y, p.z});
                        }
                    }

                    // build k-d tree
                    kdtree = std::unique_ptr<my_kd_tree_t>(new my_kd_tree_t(
                        3, cloud,
                        nanoflann::KDTreeSingleIndexAdaptorParams(10)));
                    kdtree->buildIndex();

                    // equal efficiency in image space
                    std::for_each(efficiencyList.begin(), efficiencyList.end(),
                                  [&params](PixelEfficiency &pEff) {
                                      pEff.efficiency = 1;
                                  });

#endif
                    Float effSum = std::accumulate(
                        efficiencyList.begin(), efficiencyList.end(), 0.0,
                        [](const Float &a, const PixelEfficiency &b) {
                            return a + b.efficiency;
                        });

                    uint32_t sampleCounter = 0;
                    for (size_t i = 0; i < efficiencyList.size(); ++i) {
                        auto &pEff = efficiencyList[i];
                        int ind = pEff.pixel.y * imageX + pEff.pixel.x;
                        Float ratio = pEff.efficiency / effSum;
                        pEff.sampler->SetSampleNumber(0);
                        int candidate = std::floor(SAMPLES_PER_BATCH * ratio);

                        // spp clamping
                        if (params.maxSppRatio > 0 &&
                            candidate > params.spp * params.maxSppRatio) {
                            candidate = params.spp * params.maxSppRatio;
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

                    int ind = 0;
                    for (uint32_t &i = leftovers; i > 0; --i, ++ind) {
                        if (ind >= efficiencyList.size()) ind = 0;
                        remainingSamples[efficiencyList[ind].pixel.y * imageX +
                                         efficiencyList[ind].pixel.x]++;
                    }

                    overheadTime += clock() - overheadStart;
                }
                printf("\n\n");
            }
            // reporter.Done();
        }

        globalTime = std::clock() - globalStart;

        // [Stats] ==================================================
        printf("\n------------- [Statistics] -------------\n");
        printf("Time: %fs\n", globalTime / Float(CLOCKS_PER_SEC));
        printf("Time for overhead: %fs\n",
               overheadTime / Float(CLOCKS_PER_SEC));
        printf("Time without overhead: %fs\n",
               (globalTime - overheadTime) / Float(CLOCKS_PER_SEC));
        size_t totalSample = std::accumulate(globalSampleCounter.begin(),
                                             globalSampleCounter.end(), 0);
        printf("Counted samples: %u\n", totalSample);
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

        // [Indicator] =======================================
        auto timeIndicator = std::to_string(globalTime / Float(CLOCKS_PER_SEC));
        writeText(path, timeIndicator.c_str(), std::vector<int>(), Point2i(),
                  0);
        char tmp[255];
        sprintf(tmp, "zero_(%d%%)_(%d/%d)", (counter * 100) / SAMPLES_PER_BATCH,
                counter, SAMPLES_PER_BATCH);
        writeText(path, tmp, std::vector<int>(), Point2i(), 0);

        // [Create Text] =====================================
        // writeText(path, "raynum", totalSampleNum, Point2i(256, 256), 0);
        // writeText(path, "variance", varianceMap, Point2i(256, 256), 0);
        // writeText(path, "relVariance", relVarianceMap, Point2i(256, 256),
        //          0);
        // writeText(path, "efficiency", efficiencyMap, Point2i(256, 256),
        // 0);
        writeText(path, "time", timeMap, Point2i(256, 256), 0);

        // [Create Image] =====================================
        writeImage(path, "raynum", totalSampleNum, Point2i(256, 256), 0);
        writeImage(path, "variance", varianceMap, Point2i(256, 256), 0);
        writeImage(path, "relVariance", relVarianceMap, Point2i(256, 256), 0);
        writeImage(path, "efficiency", efficiencyMap, Point2i(256, 256), 0);
        writeImage(path, "time", timeMap, Point2i(256, 256), 0);

        // Merge image tile into _Film_
        for (auto &filmTile : filmTileArray) {
            camera->film->MergeFilmTile(std::move(filmTile));
        }

        // Save final image after rendering
        camera->film->WriteImage();

        auto mse = diff2(PATH_GT, camera->film->filename);
        // printf("mse(%f), rmse(%f)\n", mse.first, mse.second);
        sprintf(tmp, "mse(%.10f),rmse(%.9f)", mse.first, mse.second);
        writeText(path, tmp, std::vector<Float>(), Point2i(), 0);
    }

    printf("----------------------------------------\n\n");
    std::cout << "Rendering finished\n\n" << std::endl;
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
