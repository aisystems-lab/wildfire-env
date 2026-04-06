import { ftToViewUnit } from "./helpers";
import { Interaction } from "../../models/ui";
import { useStores } from "../../use-stores";
import { Event } from "three";

export const useHelitackInteraction = () => {
  const { simulation, ui } = useStores();
  return {
    active: ui.interaction === Interaction.Helitack,
    onPointerDown: (e: Event) => {
      const ratio = ftToViewUnit(simulation);
      const x = e.point.x / ratio;
      const y = e.point.y / ratio;
      simulation.setHelitackPoint(x, y);
      const cell = simulation.cellAt(x, y);
      ui.interaction = null;
      console.log("Helitack", { x: x / simulation.config.modelWidth, y: y / simulation.config.modelHeight, elevation: cell.elevation });
    }
  };
};
