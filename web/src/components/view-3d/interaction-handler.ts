import { Event } from "three";

// All the events listed here:
// https://github.com/react-spring/@react-three/fiber#events
// It might be necessary to add more in the future.
export enum InteractionAction {
  // Note that onClick is intentionally not listed here. It won't work on touch devices because of OrbitControls.
  // It seems that once OrbitControls are added, they swallow onClick event generation on touch devices.
  // Probably it adds touch events and browser stops generating synthetic click events. Use onPointerDown or Up instead.
  onPointerDown = "onPointerDown",
  onPointerUp = "onPointerUp",
  onPointerOver = "onPointerOver",
  onPointerOut = "onPointerOut",
  onPointerEnter = "onPointerEnter",
  onPointerLeave = "onPointerLeave",
  onPointerMove = "onPointerMove",
  // onWheel = "onWheel" - causes type errors after updating TypeScript, and not used anyway
}

type EventHandler = (e: Event) => void;

export type InteractionHandler = {
  [action in InteractionAction]?: EventHandler;
} & {
  active: boolean
};

// Takes list of interactions and returns object with event handlers necessary for **active** interactions.
// Note that it's very important to avoid defining event handlers when no interaction is active.
// Even a single event handler triggers raycasting machinery in @react-three/fiber and it can has significant
// performance cost. Example how to use this function:
// const interactions = [ useTestInteraction(), useAnotherInteraction() ];
// const eventHandlers = getEventHandlers(interactions);
// return <mesh {...eventHandlers> /* ... */ </mesh>
export const getEventHandlers = (interactions: InteractionHandler[]) => {
  const handlers: {[action in InteractionAction]?: EventHandler[]} = {};
  interactions.forEach((interaction: InteractionHandler) => {
    if (interaction.active) {
      Object.values(InteractionAction).forEach(eventName => {
        const eventHandler = interaction[eventName];
        if (eventHandler) {
          if (!handlers[eventName]) {
            handlers[eventName] = [];
          }
          handlers[eventName]?.push(eventHandler);
        }
      });
    }
  });
  const result: {[action in InteractionAction]?: EventHandler} = {};
  Object.keys(handlers).forEach((eventName: InteractionAction) => {
    result[eventName] = (e: Event) => {
      handlers[eventName]?.forEach(handler => {
        handler(e);
      });
    };
  });
  return result;
};
