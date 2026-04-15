from ..environment.enums import DroughtLevel, TerrainType, VegetationType


class Zone:
    def __init__(
        self, vegetation=VegetationType.Barren, terrainType=TerrainType.Plains, droughtLevel=DroughtLevel.MildDrought
    ):
        self.vegetation: VegetationType = vegetation
        self.terrainType: TerrainType = terrainType
        self.droughtLevel: DroughtLevel = droughtLevel

    def __str__(self):
        return ""

    def clone(self):
        # Deepcopy if any nested mutable objects, else just recreate
        return Zone(vegetation=self.vegetation, terrainType=self.terrainType, droughtLevel=self.droughtLevel)
