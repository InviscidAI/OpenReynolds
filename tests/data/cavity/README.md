# cavity

`incompressible/icoFoam/cavity/cavity` from the OpenFOAM v2512 tutorials, meshed at
8×8×1 instead of 20×20×1 and solved, checked in as solved output.

It is here so the render tests run against a real case rather than a stand-in. pyvista
reads an OpenFOAM directory itself — no OpenFOAM installation is needed to open this,
which is why the *solved* case is committed rather than the dictionaries plus a script
to run them. 180 KB, six write times (0 to 0.5), ascii.

The x-normal cut is the useful one: `p` spans about ±1.8 over the volume and only
about -0.03 to +0.04 on that plane, so a range taken on the volume is roughly forty
times too wide. That is the mistake `--range` exists to avoid, and it needs real data
to show up.

To regenerate:

    cp -r $FOAM_TUTORIALS/incompressible/icoFoam/cavity/cavity .
    sed -i 's/(20 20 1)/(8 8 1)/' cavity/system/blockMeshDict
    cd cavity && blockMesh && icoFoam && rm -f log.*
