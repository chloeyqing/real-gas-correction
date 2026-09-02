# Real Gas Correction

A symbolic regression project for modeling deviations from the ideal gas law.

## Overview

The ideal gas law,

\[
PV = nRT
\]

provides a simple approximation for gas behavior, but it becomes inaccurate under conditions where intermolecular interactions and molecular volume are significant.

This project explores data-driven correction terms for real gases using symbolic regression, with the goal of finding compact and interpretable mathematical expressions that improve upon the ideal gas model.

## Current Approach

The project currently uses:

- Reduced temperature and pressure variables
- Experimental / reference thermodynamic data
- Symbolic regression with PySR
- Residual analysis and model comparison

## Goals

- Quantify deviations from ideal-gas behavior
- Discover interpretable correction equations
- Compare symbolic models with conventional equations of state
- Evaluate accuracy across different thermodynamic regions

## Status

Work in progress.

Current focus: symbolic regression and candidate equation evaluation.
