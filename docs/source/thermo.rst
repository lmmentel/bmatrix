Thermo
======


Input definition
----------------

INITEMPERATURE : float
    Initial temperature in K

FINTEMPERATURE : float
    Final temperature in K

TSTEP : float
    Temperature step

PRESSURE : step
    Pressure

TRANSLATIONS : str
    `YES`/`NO`

ROTATIONS : str
    `YES`/`NO`

POINTGROUP : str
    Supported point groups are:

    * `C1`
    * `Cs`
    * `C2`
    * `C2v`
    * `C3v`
    * `C2h`
    * `Coov`
    * `D2h`
    * `D3h`
    * `D5h`
    * `Dooh`
    * `D3d`
    * `Td`
    * `Oh`

PHASE : str
    Phase, one of:

    * `GP` gas phase
    * `SP` solid phase
