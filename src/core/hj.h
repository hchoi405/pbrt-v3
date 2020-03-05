#pragma once
#ifdef _WIN32
#include <windows.h>
#elif defined __linux__
#include <sys/stat.h>
#endif
#include <string>
#include <vector>
#include "imageio.h"
#include "sampler.h"

using namespace pbrt;

enum class ASMethod { Rvariance, Efficiency };

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

    // Update the mean/variance using moving average method
    void updateStats(uint32_t m, Float mMean, Float mVariance, Float mTime) {
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
        Float relativeVariance =
            variance / (pow(mean, Float(2.0)) + Float(0.01));

        // different metrics
        switch (method) {
        case ASMethod::Rvariance:
            efficiency = relativeVariance;
            break;
        case ASMethod::Efficiency:
            efficiency = relativeVariance / std::max(time, Float(1.0));
            break;
        }
    }
};

struct ExecutionResult {
    Float time;
};

struct ExecutionParams {
    int initialSpp;
    int spp;
    ASMethod method;
    Float clampThreshold;
    int maxSppRatio;
    int batchNum;
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
        sprintf(tmp, "%s_init%d_spp%d_clamp%.4f_max%d", initialSpp, spp,
                methodName.c_str(), clampThreshold, maxSppRatio);
        return std::string(tmp);
    }
};

void createDirectory(std::string path) {
#ifdef _WIN32
    if (CreateDirectory(path.c_str(), NULL) ||
        ERROR_ALREADY_EXISTS == GetLastError()) {
    } else {
        std::cout << "failed to create directory: " << path << std::endl;
    }
#else
    std::replace(path.begin(), path.end(), '\\', '/');
    if (mkdir(path.c_str(), 0777) == -1) {
        std::cout << "failed to create directory: " << path << std::endl;
    }
#endif
}

class Executor {
    std::vector<ExecutionParams> _params;
    std::vector<ExecutionResult> _results;

  public:
    Executor() { createDirectory("results\\"); }
    size_t getNum() const { return _params.size(); }

    ExecutionParams getParams(int i) { return _params[i]; }

    void addParams(ExecutionParams params) { _params.push_back(params); }

    void addResult(ExecutionResult result) { _results.push_back(result); }
};

template <typename T>
T getVariance(std::vector<T> &arr, T mean) {
    return std::accumulate(arr.begin(), arr.end(), T(0.f),
                           [&mean](const T &a, const T &b) {
                               // std::cout << a << ", " << b << std::endl;
                               return a + (b - mean) * (b - mean);
                           }) /
           arr.size();
}

template <typename T>
void writeImage(std::string path, std::string filename, std::vector<T> &values,
                Point2i res, const int OFFSET) {
    std::unique_ptr<Float[]> rgb(new Float[3 * res.x * res.y]);

    auto minmax = std::minmax_element(values.begin(), values.end());
    Float maxValue = *minmax.second;
    Float minValue = *minmax.first;
    for (int i = 0; i < res.y; ++i) {
        for (int j = 0; j < res.x; ++j) {
            int ind = i * res.x + j;
            rgb[3 * ind + 0] = Float(values[OFFSET + ind]);
            rgb[3 * ind + 1] = Float(values[OFFSET + ind]);
            rgb[3 * ind + 2] = Float(values[OFFSET + ind]);
        }
    }

    char newfilename[255];
    sprintf(newfilename, "%sstat_%s_[%.4f,%.4f].exr", path.c_str(),
            filename.c_str(), minValue, maxValue);
    pbrt::WriteImage(newfilename, &rgb[0],
                     Bounds2i(Point2i(0, 0), Point2i(res.x, res.y)),
                     Point2i(res.x, res.y));
}

template <typename T>
void writeText(std::string path, std::string filename, std::vector<T> &values,
               Point2i res, const int OFFSET) {
    char newfilename[255];

    sprintf(newfilename, "%sstat_%s.txt", path.c_str(), filename.c_str());
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
            rmse += (c0 - c1) * (c0 - c1) / (pow(c0, 2.0) + 0.01);
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

    // greater
    std::sort(begin(indices), end(indices),
              [&](size_t a, size_t b) { return values[a] > values[b]; });
    return indices;
}