import { observable, makeObservable } from "mobx";

export enum Interaction {
  PlaceSpark = "PlaceSpark",
  HoverOverDraggable = "HoverOverDraggable",
  Helitack = "Helitack"
}

export class UIModel {
  @observable public interaction: Interaction | null = null;
  @observable public dragging = false;

  constructor() {
    makeObservable(this);
  }
}
