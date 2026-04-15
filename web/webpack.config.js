'use strict';

const path = require('path');
const autoprefixer = require('autoprefixer');
const MiniCssExtractPlugin = require("mini-css-extract-plugin");
const HtmlWebpackPlugin = require('html-webpack-plugin');
const CopyWebpackPlugin = require('copy-webpack-plugin');
const webpack = require("webpack");

const WS_URL = process.env.WS_URL || "";
const GENERATED_ASSETS_BASE_URL = process.env.GENERATED_ASSETS_BASE_URL || "/data";
const GENERATED_ASSETS_DIR = path.resolve(__dirname, "../../data/generated/terrain");
const PACKAGE_DIST_DIR = path.resolve(__dirname, "../firecastrl_env/web_dist");

module.exports = (env, argv) => {
  const devMode = argv.mode !== 'production';

  return {
    context: __dirname,
    devtool: devMode ? 'eval-cheap-module-source-map' : 'source-map',
    entry: './src/index.tsx',
    mode: 'development',
    output: {
      path: PACKAGE_DIST_DIR,
      filename: 'assets/index.[contenthash].js',
    },
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
            transpileOnly: true
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
        }
      ]
    },
    resolve: {
      extensions: [ '.ts', '.tsx', '.js' ]
    },
    stats: {
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
          wsUrl: WS_URL,
          generatedAssetsBaseUrl: GENERATED_ASSETS_BASE_URL
        })
      }),
      new CopyWebpackPlugin({
        patterns: [
          { from: 'src/public' },
          { from: GENERATED_ASSETS_DIR, to: 'generated', noErrorOnMissing: true }
        ],
      }),
    ]
  };
};
