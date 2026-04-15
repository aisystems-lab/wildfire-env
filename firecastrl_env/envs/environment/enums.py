from enum import Enum


class DroughtLevel(Enum):
    NoDrought = 0
    MildDrought = 1
    MediumDrought = 2
    SevereDrought = 3


class VegetationType(Enum):
    EvergreenNeedleleaf = 1
    EvergreenBroadleaf = 2
    DeciduousNeedleleaf = 3
    DeciduousBroadleaf = 4
    MixedForest = 5
    ClosedShrublands = 6
    OpenShrublands = 7
    WoodySavannas = 8
    Savannas = 9
    Grasslands = 10
    PermanentWetlands = 11
    Croplands = 12
    UrbanBuilt = 13
    CroplandMosaic = 14
    SnowIce = 15
    Barren = 16
    Water = 17


class TerrainType(Enum):
    Mountains = "mountains"
    Plains = "plains"
    Hills = "hills"
    Tropical = "tropical"
    Desert = "desert"
    Wetlands = "wetlands"
    Agricultural = "agricultural"
    Urban = "urban"
    Mixed = "mixed"
    Ice = "ice"
    Water = "water"


class FireState:
    Unburnt = 0
    Burning = 1
    Burnt = 2


class BurnIndex:
    Low = 0
    Medium = 1
    High = 2
