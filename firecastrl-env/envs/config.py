import math
from environment.enums import DroughtLevel, VegetationType, TerrainType

# Environment settings
MAX_TIMESTEPS = 1000
HELICOPTER_SPEED = 3
SEED = 42

modelWidth= 120000
modelHeight= 80000
gridWidth = 240
cellSize = modelWidth / gridWidth
gridHeight = math.ceil(modelHeight / cellSize)
maxTimeStep= 180 # minutes
modelDayInSeconds= 8 # one day in model should last X seconds in real world
windSpeed= 0 # mph
windDirection= 0 # degrees northern wind
neighborsDist= 2.5
minCellBurnTime= 200 # minutes
heightmapMaxElevation= 10000

zonesCount= 18
RGB_TO_LAND_COVER_INDEX = {
    "0,100,0": 1,       # Evergreen Needleleaf Forest
    "34,139,34": 2,     # Evergreen Broadleaf Forest
    "50,205,50": 3,     # Deciduous Needleleaf Forest
    "0,128,0": 4,       # Deciduous Broadleaf Forest
    "60,179,113": 5,    # Mixed Forest
    "240,230,140": 6,   # Closed Shrublands
    "218,165,32": 7,    # Open Shrublands
    "128,128,0": 8,     # Woody Savannas
    "154,205,50": 9,    # Savannas
    "144,238,144": 10,  # Grasslands
    "143,188,143": 11,  # Permanent Wetlands
    "210,180,140": 12,  # Croplands
    "128,128,128": 13,  # Urban and Built-Up
    "222,184,135": 14,  # Cropland/Natural Vegetation Mosaic
    "255,255,255": 15,  # Snow and Ice
    "211,211,211": 16,  # Barren or Sparsely Vegetated
    "0,0,255": 17       # Water
}
helitackDropRadius= 2640 # ft (5280 ft = 1 mile)
fireSurvivalProbability= 0.1
ZONES = [
    {
        "terrainType": TerrainType.Mountains,
        "vegetation": VegetationType.EvergreenNeedleleaf,
        "droughtLevel": DroughtLevel.MildDrought  # 0. Dummy Zone
    },
    {
        "terrainType": TerrainType.Mountains,
        "vegetation": VegetationType.EvergreenNeedleleaf,
        "droughtLevel": DroughtLevel.MildDrought  # 1. Evergreen Needleleaf Forest
    },
    {
        "terrainType": TerrainType.Tropical,
        "vegetation": VegetationType.EvergreenBroadleaf,
        "droughtLevel": DroughtLevel.NoDrought  # 2. Evergreen Broadleaf Forest
    },
    {
        "terrainType": TerrainType.Plains,
        "vegetation": VegetationType.DeciduousNeedleleaf,
        "droughtLevel": DroughtLevel.MildDrought  # 3. Deciduous Needleleaf Forest
    },
    {
        "terrainType": TerrainType.Plains,
        "vegetation": VegetationType.DeciduousBroadleaf,
        "droughtLevel": DroughtLevel.MildDrought  # 4. Deciduous Broadleaf Forest
    },
    {
        "terrainType": TerrainType.Hills,
        "vegetation": VegetationType.MixedForest,
        "droughtLevel": DroughtLevel.MediumDrought  # 5. Mixed Forest
    },
    {
        "terrainType": TerrainType.Desert,
        "vegetation": VegetationType.ClosedShrublands,
        "droughtLevel": DroughtLevel.MediumDrought  # 6. Closed Shrublands
    },
    {
        "terrainType": TerrainType.Desert,
        "vegetation": VegetationType.OpenShrublands,
        "droughtLevel": DroughtLevel.SevereDrought  # 7. Open Shrublands
    },
    {
        "terrainType": TerrainType.Plains,
        "vegetation": VegetationType.WoodySavannas,
        "droughtLevel": DroughtLevel.MildDrought  # 8. Woody Savannas
    },
    {
        "terrainType": TerrainType.Plains,
        "vegetation": VegetationType.Savannas,
        "droughtLevel": DroughtLevel.MediumDrought  # 9. Savannas
    },
    {
        "terrainType": TerrainType.Plains,
        "vegetation": VegetationType.Grasslands,
        "droughtLevel": DroughtLevel.MildDrought  # 10. Grasslands
    },
    {
        "terrainType": TerrainType.Wetlands,
        "vegetation": VegetationType.PermanentWetlands,
        "droughtLevel": DroughtLevel.NoDrought  # 11. Permanent Wetlands
    },
    {
        "terrainType": TerrainType.Agricultural,
        "vegetation": VegetationType.Croplands,
        "droughtLevel": DroughtLevel.MildDrought  # 12. Croplands
    },
    {
        "terrainType": TerrainType.Urban,
        "vegetation": VegetationType.UrbanBuilt,
        "droughtLevel": DroughtLevel.SevereDrought  # 13. Urban and Built-Up
    },
    {
        "terrainType": TerrainType.Mixed,
        "vegetation": VegetationType.CroplandMosaic,
        "droughtLevel": DroughtLevel.MediumDrought  # 14. Cropland/Natural Vegetation Mosaic
    },
    {
        "terrainType": TerrainType.Ice,
        "vegetation": VegetationType.SnowIce,
        "droughtLevel": DroughtLevel.NoDrought  # 15. Snow and Ice
    },
    {
        "terrainType": TerrainType.Desert,
        "vegetation": VegetationType.Barren,
        "droughtLevel": DroughtLevel.SevereDrought  # 16. Barren or Sparsely Vegetated
    },
    {
        "terrainType": TerrainType.Water,
        "vegetation": VegetationType.Water,
        "droughtLevel": DroughtLevel.NoDrought  # 17. Water
    },
]

MAX_BURN_TIME = 500
fillTerrainEdges= True