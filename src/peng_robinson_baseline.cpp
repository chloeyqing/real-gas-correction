// Direct p(T, molar density) Peng-Robinson baseline; no pressure-to-density flash.
// Coefficients match CoolProp 8.0.0 GeneralizedCubic.cpp, PengRobinson methods.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr double R = 8.31446261815324;
constexpr double TC = 304.1282;
constexpr double PC = 7377300.0;
constexpr double OMEGA = 0.22394;

double pressure(double temperature, double density) {
    const double a = 0.45723552892138218938 * R * R * TC * TC / PC;
    const double b = 0.07779607390388455972 * R * TC / PC;
    const double kappa = 0.37464 + 1.54226 * OMEGA - 0.26992 * OMEGA * OMEGA;
    const double v = 1.0 / density;
    if (!(temperature > 0 && density > 0 && v > b))
        throw std::domain_error("Input outside the positive-volume PR domain");
    const double alpha_root = 1.0 + kappa * (1.0 - std::sqrt(temperature / TC));
    return R * temperature / (v - b)
        - a * alpha_root * alpha_root / (v * (v + b) + b * (v - b));
}

std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string value;
    while (std::getline(stream, value, ',')) fields.push_back(value);
    return fields;
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Usage: peng_robinson_baseline input.csv");
        std::ifstream input(argv[1]);
        if (!input) throw std::runtime_error("Cannot open input CSV");
        std::string line;
        if (!std::getline(input, line)) throw std::runtime_error("Empty CSV");
        const auto header = split(line);
        auto column = [&](const std::string& name) {
            const auto pos = std::find(header.begin(), header.end(), name);
            if (pos == header.end()) throw std::runtime_error("Missing column: " + name);
            return static_cast<std::size_t>(pos - header.begin());
        };
        const auto ti = column("T_K"), ri = column("rho_mol_m3");
        std::cout << "T_K,rho_mol_m3,p_PR_Pa\n" << std::setprecision(17);
        while (std::getline(input, line)) {
            if (line.empty()) continue;
            const auto row = split(line);
            if (row.size() != header.size()) throw std::runtime_error("Malformed CSV row");
            const double t = std::stod(row.at(ti)), rho = std::stod(row.at(ri));
            const double p = pressure(t, rho);
            if (!(std::isfinite(p) && p > 0)) throw std::runtime_error("Invalid PR pressure");
            std::cout << t << ',' << rho << ',' << p << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
