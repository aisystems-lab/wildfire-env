import math
from typing import Optional

from ..environment.enums import BurnIndex, DroughtLevel, FireState, VegetationType
from ..environment.zone import Zone

FIRE_LINE_DEPTH = 2000
MAX_BURN_TIME = 500

moisture_lookup_by_land_cover: dict[VegetationType, list[float]] = {
    VegetationType.EvergreenNeedleleaf: [0.30, 0.25, 0.20, 0.15],
    VegetationType.EvergreenBroadleaf: [0.32, 0.27, 0.22, 0.17],
    VegetationType.DeciduousNeedleleaf: [0.28, 0.23, 0.18, 0.13],
    VegetationType.DeciduousBroadleaf: [0.30, 0.25, 0.20, 0.15],
    VegetationType.MixedForest: [0.29, 0.24, 0.19, 0.14],
    VegetationType.ClosedShrublands: [0.25, 0.20, 0.15, 0.10],
    VegetationType.OpenShrublands: [0.22, 0.17, 0.12, 0.08],
    VegetationType.WoodySavannas: [0.24, 0.19, 0.14, 0.10],
    VegetationType.Savannas: [0.21, 0.17, 0.13, 0.09],
    VegetationType.Grasslands: [0.20, 0.16, 0.12, 0.08],
    VegetationType.PermanentWetlands: [0.35, 0.30, 0.25, 0.20],
    VegetationType.Croplands: [0.23, 0.18, 0.13, 0.09],
    VegetationType.CroplandMosaic: [0.22, 0.17, 0.12, 0.08],
    VegetationType.UrbanBuilt: [0.10, 0.08, 0.06, 0.04],
    VegetationType.Barren: [0.05, 0.04, 0.03, 0.02],
    VegetationType.SnowIce: [0.00, 0.00, 0.00, 0.00],
    VegetationType.Water: [0.00, 0.00, 0.00, 0.00],
}


class Cell:
    def __init__(
        self,
        *,
        x: int,
        y: int,
        zone: Zone,
        zoneIdx: Optional[int] = None,
        baseElevation: float = 0.0,
        ignitionTime: float = math.inf,
        fireState: FireState = FireState.Unburnt,
        isUnburntIsland: bool = False,
        isRiver: bool = False,
        isFireLine: bool = False,
        isFireLineUnderConstruction: bool = False,
    ):

        self.x = x
        self.y = y
        self.zone: Zone = zone
        self.zoneIdx = zoneIdx if zoneIdx is not None else 0
        self.baseElevation = baseElevation
        self.ignitionTime = ignitionTime
        self.spreadRate = 0.0
        self.burnTime = MAX_BURN_TIME
        self.fireState = fireState
        self.isUnburntIsland = isUnburntIsland
        self.isRiver = isRiver
        self.isFireLine = isFireLine
        self.isFireLineUnderConstruction = isFireLineUnderConstruction
        self.helitackDropCount = 0
        self.isFireSurvivor = False

    def __str__(self):
        """
        Returns a human-readable string representation of the FireCell object.
        This method is automatically called when you use print() on an object.
        """
        return (
            f"FireCell at ({self.x}, {self.y}):\n"
            f"  - Zone: {self.zone} (Index: {self.zoneIdx})\n"
            f"  - Base Elevation: {self.baseElevation}\n"
            f"  - Ignition Time: {self.ignitionTime}\n"
            f"  - Spread Rate: {self.spreadRate}\n"
            f"  - Burn Time: {self.burnTime}\n"
            f"  - Fire State: {self.fireState}\n"
            f"  - Is Unburnt Island: {self.isUnburntIsland}\n"
            f"  - Is River: {self.isRiver}\n"
            f"  - Is Fire Line: {self.isFireLine}\n"
            f"  - Is Fire Line Under Construction: {self.isFireLineUnderConstruction}\n"
            f"  - Helitack Drop Count: {self.helitackDropCount}\n"
            f"  - Is Fire Survivor: {self.isFireSurvivor}"
        )

    @property
    def vegetation(self) -> VegetationType:
        return self.zone.vegetation

    @property
    def elevation(self) -> float:
        if self.isFireLine:
            return self.baseElevation - FIRE_LINE_DEPTH
        return self.baseElevation

    @property
    def isNonburnable(self) -> bool:
        return self.isRiver or self.isUnburntIsland

    @property
    def droughtLevel(self) -> DroughtLevel:
        if self.helitackDropCount > 0:
            new_level = self.zone.droughtLevel.value - self.helitackDropCount
            clamped_level = max(new_level, DroughtLevel.NoDrought.value)
            return DroughtLevel(clamped_level)
        return self.zone.droughtLevel

    @property
    def moistureContent(self) -> float:
        if self.isNonburnable:
            return math.inf
        return moisture_lookup_by_land_cover[self.vegetation][self.droughtLevel.value]

    @property
    def isBurningOrWillBurn(self) -> bool:
        return self.fireState == FireState.Burning or (
            self.fireState == FireState.Unburnt and self.ignitionTime < math.inf
        )

    @property
    def canSurviveFire(self) -> bool:
        return self.burnIndex == BurnIndex.Low and self.vegetation == VegetationType.PermanentWetlands

    @property
    def burnIndex(self) -> BurnIndex:
        veg = self.vegetation
        sr = self.spreadRate

        if veg == VegetationType.Grasslands:
            return BurnIndex.Low if sr < 45 else BurnIndex.Medium

        if veg in (VegetationType.ClosedShrublands, VegetationType.OpenShrublands):
            if sr < 10:
                return BurnIndex.Low
            if sr < 50:
                return BurnIndex.Medium
            return BurnIndex.High

        if veg in (
            VegetationType.EvergreenNeedleleaf,
            VegetationType.DeciduousNeedleleaf,
            VegetationType.MixedForest,
            VegetationType.EvergreenBroadleaf,
            VegetationType.DeciduousBroadleaf,
        ):
            return BurnIndex.Low if sr < 25 else BurnIndex.Medium

        # Default fallback:
        return BurnIndex.Low

    def isBurnableForBI(self, burnIndex: BurnIndex) -> bool:
        # Fire lines will burn only if burnIndex is High
        return not self.isNonburnable and (not self.isFireLine or burnIndex == BurnIndex.High)

    def reset(self):
        self.ignitionTime = math.inf
        self.spreadRate = 0.0
        self.burnTime = MAX_BURN_TIME
        self.fireState = FireState.Unburnt
        self.isFireLineUnderConstruction = False
        self.isFireLine = False
        self.helitackDropCount = 0
        self.isFireSurvivor = False
