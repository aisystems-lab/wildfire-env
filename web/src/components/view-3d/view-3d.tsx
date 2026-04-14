import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { Provider } from "mobx-react";
import React, { useRef } from "react";
import * as THREE from "three";
import { useStores } from "../../use-stores";
import { FireLineMarkersContainer } from "./fire-line-marker";
import { DEFAULT_UP, PLANE_WIDTH, planeHeight } from "./helpers";
import { HelicopterMarker } from "./helicopter-marker";
import { SparksContainer } from "./spark";
import { Terrain } from "./terrain";

export const View3d = () => {
  const stores = useStores();
  const simulation = stores.simulation;
  const ui = stores.ui;
  const cameraPos: [number, number, number] = [PLANE_WIDTH * 0.5, planeHeight(simulation) * -1.5, PLANE_WIDTH * 1.5];
  const terrainRef = useRef<THREE.Mesh>(null);

  return (
    /* eslint-disable react/no-unknown-property */
    // See: https://github.com/jsx-eslint/eslint-plugin-react/issues/3423
    // flat=true disables tone mapping that is not a default in threejs, but is enabled by default in react-three-fiber.
    // It makes textures match colors in the original image.
    <Canvas
      flat={true}
    // // ① expose alpha & antialias on the WebGLRenderer
    // gl={{ antialias: true, alpha: true }}
    // // ② run code immediately after the renderer+scene are created
    // onCreated={({ gl, scene }) => {
    //   // make the clear color fully transparent
    //   gl.setClearColor(0x000000, 0);

    //   // remove the default gray background
    //   scene.background = null;

    //   // load a simple cube‐map sky (6 images in /public/textures/)
    //   const loader = new THREE.CubeTextureLoader();
    //   const envMap = loader.load([
    //     '/textures/sky_px.jpg',
    //     '/textures/sky_nx.jpg',
    //     '/textures/sky_py.jpg',
    //     '/textures/sky_ny.jpg',
    //     '/textures/sky_pz.jpg',
    //     '/textures/sky_nz.jpg',
    //   ]);

    //   // assign it as both background and environment
    //   scene.environment = envMap;
    //   scene.background  = envMap;
    // }}
    >
      {/* Why do we need to setup provider again? No idea. It seems that components inside Canvas don't have
          access to MobX stores anymore. */}
      <Provider stores={stores}>
        <PerspectiveCamera makeDefault={true} fov={33} position={cameraPos} up={DEFAULT_UP} />
        <OrbitControls
          target={[PLANE_WIDTH * 0.5, planeHeight(simulation) * 0.5, 0.2]}
          enableDamping={true}
          enableRotate={!ui.dragging} // disable rotation when something is being dragged
          enablePan={false}
          rotateSpeed={0.5}
          zoomSpeed={0.5}
          minDistance={0.8}
          maxDistance={5}
          maxPolarAngle={Math.PI * 0.4}
          minAzimuthAngle={-Math.PI * 0.25}
          maxAzimuthAngle={Math.PI * 0.25}
        />
        <hemisphereLight args={[0xC6C2B6, 0x3A403B, 1.2]} up={DEFAULT_UP} />
        <directionalLight
          args={[0xffffff, 0.8]}
          position={[-100, 100, -100]}
        />
        <Terrain ref={terrainRef} />
        <HelicopterMarker />
        <SparksContainer dragPlane={terrainRef} />
        <FireLineMarkersContainer dragPlane={terrainRef} />
      </Provider>
    </Canvas>
    /* eslint-enable react/no-unknown-property */
  );
};
