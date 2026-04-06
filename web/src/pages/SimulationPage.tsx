import { observer } from "mobx-react";
import React, { useEffect } from "react";
import { View3d } from "../components/view-3d/view-3d";
import { useStores } from "../use-stores";
import css from "../components/app.module.scss";
import { useCustomCursor } from "../components/use-custom-cursors";

export const SimulationPage = observer(function WrappedComponent() {
  const { simulation } = useStores();

  useEffect(() => {
  // Connect WebSocket only once when the page loads
  simulation.connectSocket();

  return () => {
    simulation.cleanup(); 
  };
}, []);

  // This will setup document cursor based on various states of UI store (interactions).
  useCustomCursor();

  const config = simulation.config;
  // Convert time from minutes to days.
  const timeInDays = Math.floor(simulation.time / 1440);
  const timeHours = Math.floor((simulation.time % 1440) / 60);
  const showModelScale = config.showModelDimensions;
  const episodeCount = simulation.episodeCount;
  return (
    <div className={css.app}>
      { showModelScale &&
        <div className={css.modelInfo}>
          <div>Model Dimensions: { config.modelWidth } ft x { config.modelHeight } ft</div>
          <div>Highest Point Possible: {config.heightmapMaxElevation} ft</div>
        </div>
      }
      <div className={css.metricsDisplay}>
        <div>Step: {simulation.stepCount}</div>
        <div>Burning: {simulation.cellsBurning}</div>
        <div>Burnt: {simulation.cellsBurnt}</div>
        <div>Heli: [{simulation.helicopterGridCoord[0]}, {simulation.helicopterGridCoord[1]}]</div>
      </div>
      <div className={css.timeDisplay}>
        <div>Episode: {episodeCount}</div>
        <div>{timeInDays} {timeInDays === 1 ? "day" : "days"}</div>
        <div>{timeHours} {timeHours === 1 ? "hour" : "hours"}</div>
      </div>
      <div className={css.mainContent}>
        <View3d />
      </div>
    </div>
  );
});
