package com.lohika.morning.ml.spark.distributed.library.function.map.generic;

import java.util.Arrays;

public class DoubleArrayVarianceHolder {

    private Double label;
    private double[] averages;
    private double[][] featureMinusAverages;
    private int trainingSetCounts;

    public DoubleArrayVarianceHolder() {
    }

    public DoubleArrayVarianceHolder(int size) {
        this.featureMinusAverages = new double[size*size][size];
        this.trainingSetCounts = -1;
    }

    public Double getLabel() {
        return label;
    }

    public void setLabel(Double label) {
        this.label = label;
    }

    public double[] getAverages() {
        return averages;
    }

    public void setAverages(double[] averages) {
        this.averages = averages;
    }

    public double[][] getFeatureMinusAverages() {
        return featureMinusAverages;
    }

    public void setFeatureMinusAverages(double[][] featureMinusAverages) {
        this.featureMinusAverages = featureMinusAverages;
    }

    public int getTrainingSetCounts() {
        return trainingSetCounts;
    }

    public void setTrainingSetCounts(int trainingSetCounts) {
        this.trainingSetCounts = trainingSetCounts;
    }

    @Override
    public String toString() {
        return "DoubleArrayAVGHolder{" +
                "label=" + label +
                ", featureMinusAverages=" + Arrays.toString(featureMinusAverages) +
                ", trainingSetCounts=" + trainingSetCounts +
                '}';
    }

}
