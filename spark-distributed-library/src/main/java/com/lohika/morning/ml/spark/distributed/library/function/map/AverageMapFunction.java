package com.lohika.morning.ml.spark.distributed.library.function.map;

import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;

public class AverageMapFunction implements MapFunction<Row, Row> {

    @Override
    public Row call(Row input) throws Exception {
        DenseVector vector = (DenseVector) input.getAs("features");
        double sum = 0D;
        for (int i = 0; i < vector.values().length; i++) {
            sum = sum + vector.values()[i];
        }

        double averagePerWord = sum/vector.values().length;

        double[] delta = new double[vector.values().length];
        for (int i = 0; i < vector.values().length; i++) {
            delta[i] = vector.values()[i] - averagePerWord;
        }

        double[][] cov = multiply(delta, delta);

        return RowFactory.create(input.getAs("value"),
                                 input.getAs("label"),
                                 input.getAs("id"),
                                 input.getAs("words"),
                                 input.getAs("features"), averagePerWord, Vectors.dense(delta), cov);
    }

    public static double[][] multiply(double[] a, double[] b) {
        double[][] c = new double[a.length][b.length];
        for (int i = 0; i < a.length; i++)
            for (int j = 0; j < b.length; j++)
                    c[i][j] = a[i] * b[j];
        return c;
    }

}
