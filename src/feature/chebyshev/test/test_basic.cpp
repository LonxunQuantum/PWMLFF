#include "basic.h"

class test_basic {
public:
    test_basic(int size, double rcut_max, double rcut_smooth)
        : chebyshev(size, rcut_max, rcut_smooth) {}

    void build(double rij) {
        chebyshev.build(rij);
    }

    void show() const {
        chebyshev.show();
    }

private:
    Chebyshev1st<double> chebyshev;
};

int main() {
    test_basic tb(10, 3.2, 0.5);
    tb.build(2.0);
    tb.show();
    return 0;
}