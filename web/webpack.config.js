'use strict';

const path = require('path');
const autoprefixer = require('autoprefixer');
const MiniCssExtractPlugin = require("mini-css-extract-plugin");
const HtmlWebpackPlugin = require('html-webpack-plugin');
const CopyWebpackPlugin = require('copy-webpack-plugin');
const webpack = require("webpack");

// DEPLOY_PATH is set by the s3-deploy-action its value will be:
// `branch/[branch-name]/` or `version/[tag-name]/`
// See the following documentation for more detail:
//   https://github.com/concord-consortium/s3-deploy-action/blob/main/README.md#top-branch-example
const DEPLOY_PATH = process.env.DEPLOY_PATH;
const API_BASE_URL = process.env.API_BASE_URL || "http://localhost:6969";
const WS_URL = process.env.WS_URL || "";
const GENERATED_ASSETS_BASE_URL = process.env.GENERATED_ASSETS_BASE_URL || "/data";
const GENERATED_ASSETS_DIR = path.resolve(__dirname, "../../data/generated/terrain");
const PACKAGE_DIST_DIR = path.resolve(__dirname, "../firecastrl_env/web_dist");

module.exports = (env, argv) => {
  const devMode = argv.mode !== 'production';

  return {
    context: __dirname, // to automatically find tsconfig.json
    devtool: devMode ? 'eval-cheap-module-source-map' : 'source-map',
    entry: './src/index.tsx',
    mode: 'development',
    output: {
      path: PACKAGE_DIST_DIR,
      filename: 'assets/index.[contenthash].js',
    },
    // Add this devServer configuration
    devServer: {
      allowedHosts: 'all',
      host: '0.0.0.0',
      port: 8080,
      historyApiFallback: true,
      static: [
        {
          directory: path.resolve(__dirname, "src/public"),
          publicPath: "/"
        },
        {
          directory: GENERATED_ASSETS_DIR,
          publicPath: "/generated"
        }
      ],
      client: {
        webSocketURL: 'auto://0.0.0.0:0/ws'
      },
      headers: {
        "Access-Control-Allow-Origin": "*"
      }
    },
    performance: { hints: false },
    module: {
      rules: [
        {
          test: /\.tsx?$/,
          loader: 'ts-loader',
          options: {
            transpileOnly: true // IMPORTANT! use transpileOnly mode to speed-up compilation
          }
        },
        {
          test: /\.(sa|sc|le)ss$/i,
          use: [
            devMode ? 'style-loader' : MiniCssExtractPlugin.loader,
            {
              loader: 'css-loader',
              options: {
                modules: {
                  localIdentName: '[name]--[local]--__wildfire-v1__'
                },
                sourceMap: true,
                importLoaders: 1
              }
            },
            {
              loader: 'postcss-loader',
              options: {
                postcssOptions: {
                  plugins: [autoprefixer()]
                }
              }
            },
            'sass-loader'
          ]
        },
        {
          test: /\.css$/i,
          use: [
            devMode ? 'style-loader' : MiniCssExtractPlugin.loader,
            'css-loader'
          ]
        },
        {
          test: /\.(png|woff|woff2|eot|ttf)$/,
          type: 'asset',
        },
        {
          test: /\.svg$/i,
          exclude: /\.nosvgo\.svg$/i,
          oneOf: [
            {
              // Do not apply SVGR import in CSS files.
              issuer: /\.(css|scss|less)$/,
              type: 'asset',
            },
            {
              issuer: /\.tsx?$/,
              loader: '@svgr/webpack',
              options: {
                svgoConfig: {
                  plugins: [
                    {
                      // cf. https://github.com/svg/svgo/releases/tag/v2.4.0
                      name: 'preset-default',
                      params: {
                        overrides: {
                          // don't minify "id"s (i.e. turn randomly-generated unique ids into "a", "b", ...)
                          // https://github.com/svg/svgo/blob/master/plugins/cleanupIds.js
                          cleanupIds: { minify: false },
                          // leave <line>s, <rect>s and <circle>s alone
                          // https://github.com/svg/svgo/blob/master/plugins/convertShapeToPath.js
                          convertShapeToPath: false,
                          // leave "stroke"s and "fill"s alone
                          // https://github.com/svg/svgo/blob/master/plugins/removeUnknownsAndDefaults.js
                          removeUnknownsAndDefaults: { defaultAttrs: false },
                          // leave viewBox alone
                          removeViewBox: false
                        }
                      }
                    }
                  ]
                }
              }
            }
          ]
        }
      ]
    },
    resolve: {
      extensions: [ '.ts', '.tsx', '.js' ]
    },
    stats: {
      // suppress "export not found" warnings about re-exported types
      warningsFilter: /export .* was not found in/
    },
    plugins: [
      new MiniCssExtractPlugin({
        filename: devMode ? 'assets/[name].css' : 'assets/[name].[contenthash].css',
      }),
      new HtmlWebpackPlugin({
        filename: 'index.html',
        template: 'src/index.html',
        favicon: 'src/public/favicon.ico'
      }),
      new webpack.DefinePlugin({
        __APP_CONFIG__: JSON.stringify({
          apiBaseUrl: API_BASE_URL,
          wsUrl: WS_URL,
          generatedAssetsBaseUrl: GENERATED_ASSETS_BASE_URL
        })
      }),
      ...(DEPLOY_PATH ? [new HtmlWebpackPlugin({
        filename: 'index-top.html',
        template: 'src/index.html',
        favicon: 'src/public/favicon.ico',
        publicPath: DEPLOY_PATH
      })] : []),
      new CopyWebpackPlugin({
        patterns: [
          { from: 'src/public' },
          { from: GENERATED_ASSETS_DIR, to: 'generated', noErrorOnMissing: true }
        ],
      }),
    ]
  };
};
