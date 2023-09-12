
//                                                _ _ _                          
//   _____   _____ _ __ _ __   __ _ _ __   ___   | (_) |__  _ __ __ _ _ __ _   _ 
//  / _ \ \ / / _ \ '__| '_ \ / _` | '_ \ / _ \  | | | '_ \| '__/ _` | '__| | | |
// |  __/\ V /  __/ |  | |_) | (_| | | | | (_) | | | | |_) | | | (_| | |  | |_| |
//  \___| \_/ \___|_|  | .__/ \__,_|_| |_|\___/  |_|_|_.__/|_|  \__,_|_|   \__, |
//                     |_|                                                 |___/ 




// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const path = require('path');
const CopyPlugin = require("copy-webpack-plugin");
const webpack = require('webpack');
// const TerserPlugin = require("terser-webpack-plugin");

module.exports = () => {
    return {
        target: ['web'],
        entry: path.resolve(__dirname, '/src/app.js'),
        output: {
            path: path.resolve(__dirname, 'dist'),
            filename: 'everpano.js',
            // library: {
            //     type: 'umd'
            // },
            libraryTarget: 'var',
            library: 'everpano'
        },
        plugins: [
            new webpack.DefinePlugin({
                'process.browser': 'true'
            }),
            new CopyPlugin({
            // Use copy plugin to copy *.wasm to output folder.
            patterns: [{ from: 'node_modules/onnxruntime-web/dist/*.wasm', to: '[name][ext]' }]
        })],

        // {
        //     plugins: [
        //       new webpack.DefinePlugin({
        //         'process.browser': 'true'
        //       }),
        //       ...
        //     ],
        //   }
        mode: 'production'
    }
};