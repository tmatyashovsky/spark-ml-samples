package com.lohika.morning.ml.spark.distributed.library.function.map.generic;

import java.util.Arrays;

public class DoubleArrayAVGHolder {

    private Double label;
    private double[] sums;
    private int[] counts;

    public DoubleArrayAVGHolder() {
    }

    public DoubleArrayAVGHolder(int size) {
        this.sums = new double[size];
        this.counts = new int[size];
    }

    public DoubleArrayAVGHolder(Double label, double[] sums, int[] counts) {
        this.label = label;
        this.sums = sums;
        this.counts = counts;
    }

    public Double getLabel() {
        return label;
    }

    public void setLabel(Double label) {
        this.label = label;
    }

    public double[] getSums() {
        return sums;
    }

    public void setSums(double[] sums) {
        this.sums = sums;
    }

    public int[] getCounts() {
        return counts;
    }

    public void setCounts(int[] counts) {
        this.counts = counts;
    }

    @Override
    public String toString() {
        return "DoubleArrayAVGHolder{" +
                "label=" + label +
                ", sums=" + Arrays.toString(sums) +
                ", counts=" + Arrays.toString(counts) +
                '}';
    }

}
