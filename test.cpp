#include <optional>
#include <iostream>

int main() {
    std::optional<int> opt;
    if (!opt) {
        std::cout << "C++17 is enabled!\n";
    }
    return 0;
}
