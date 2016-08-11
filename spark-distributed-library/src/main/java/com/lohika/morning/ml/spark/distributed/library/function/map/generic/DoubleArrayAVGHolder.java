package com.lohika.morning.ml.spark.distributed.library.function.map.generic;

import java.util.Arrays;

public class DoubleArrayAVGHolder {

    private Double label;
    private double[] averageSums;
    private int counts;
    private double[][] covarianceSums;

    public DoubleArrayAVGHolder() {
    }

    public DoubleArrayAVGHolder(int size) {
        this.averageSums = new double[size];
        this.covarianceSums = new double[size][size];
        this.counts = 0;
    }

    public DoubleArrayAVGHolder(Double label, double[] averageSums, int counts, double[][] covarianceSums) {
        this.label = label;
        this.averageSums = averageSums;
        this.counts = counts;
        this.covarianceSums = covarianceSums;
    }

    public Double getLabel() {
        return label;
    }

    public void setLabel(Double label) {
        this.label = label;
    }

    public double[] getAverageSums() {
        return averageSums;
    }

    public void setAverageSums(double[] averageSums) {
        this.averageSums = averageSums;
    }

    public int getCounts() {
        return counts;
    }

    public void setCounts(int counts) {
        this.counts = counts;
    }

    public double[][] getCovarianceSums() {
        return covarianceSums;
    }

    public void setCovarianceSums(double[][] covarianceSums) {
        this.covarianceSums = covarianceSums;
    }

    @Override
    public String toString() {
        return "DoubleArrayAVGHolder{" +
                "label=" + label +
                ", averageSums=" + Arrays.toString(averageSums) +
                ", counts=" + counts +
                '}';
    }

}
