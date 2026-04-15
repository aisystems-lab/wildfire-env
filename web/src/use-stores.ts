import { useContext } from "react";
import { MobXProviderContext } from "mobx-react";
import { IStores } from "./models/stores";

export const useStores = (): IStores => {
  return useContext(MobXProviderContext).stores;
};
