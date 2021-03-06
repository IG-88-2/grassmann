const path = require("path");
const CopyPlugin = require("copy-webpack-plugin");
const HtmlWebpackPlugin = require('html-webpack-plugin');
const WasmPackPlugin = require("@wasm-tool/wasm-pack-plugin");

const appConfig = {
  mode: "development",
  entry: "./main.ts",
  output: {
    filename: '[name].js',
    path: path.resolve(__dirname, "development")
  },
  target:"web",
  module: {
    rules:[
      {
        test: /\.(ts)?$/,
        exclude: path.resolve(__dirname, 'node_modules'),
        loader: "ts-loader"
      }
    ]
  },
  devServer: {
    port: 3000,
    hot: true,
    historyApiFallback: true
  },
  resolve: {
    extensions: [".ts", ".js", ".wasm"]
  },
  experiments: {
    syncWebAssembly: true
  },
  plugins: [
    new HtmlWebpackPlugin(),
    new WasmPackPlugin({
      crateDirectory: path.resolve(__dirname, '.'),
      outDir: "out",
      outName: "grassmann"
    }),
    new CopyPlugin({
      patterns: [
        { 
          from: "./out", 
          to: path.resolve(__dirname, "development")
        }
      ]
    })
  ]
};

const workerConfig = {
  mode: "development",
  entry: "./worker.ts",
  target: "webworker",
  module: {
    rules:[
      {
        test: /\.(ts)?$/,
        exclude: path.resolve(__dirname, 'node_modules'),
        loader: "ts-loader"
      }
    ]
  },
  resolve: {
    extensions: [".ts", ".js", ".wasm"]
  },
  experiments: {
    syncWebAssembly: true
  },
  output: {
    filename: 'worker.js',
    path: path.resolve(__dirname, "development")
  }
};

module.exports = [appConfig, workerConfig];
