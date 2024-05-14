#include "radial.h"

using namespace std;

class test_radial {
public:
    test_radial(int miu, int size, int ntypes, double rcut_max, double rcut_smooth)
        : radial(miu, size, ntypes, rcut_max, rcut_smooth) {}

    void build(double rij, int itype, int jtype) {
        radial.build(rij, itype, jtype);
    }

    void show() const {
        radial.show();
    }

private:
    Radial<double> radial;
};

int main() {
    test_radial tb(25, 10, 2, 3.2, 0.5);
    tb.build(2.0, 0, 0);
    tb.show();
    return 0;
}
