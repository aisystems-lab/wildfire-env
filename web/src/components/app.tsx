import { observer } from "mobx-react";
import React from "react";

import { SimulationPage } from "../pages/SimulationPage";
import styles from "./app.module.scss";

export const AppComponent = observer(function WrappedComponent() {
  return (
    <div id={styles.app}>
      <SimulationPage />
    </div>
  );
});
