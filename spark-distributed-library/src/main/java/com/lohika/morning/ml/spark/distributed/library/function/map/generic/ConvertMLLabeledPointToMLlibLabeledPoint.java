package com.lohika.morning.ml.spark.distributed.library.function.map.generic;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Row;

public class ConvertMLLabeledPointToMLlibLabeledPoint implements Function<Row, LabeledPoint> {

    private boolean includeVariances;

    public ConvertMLLabeledPointToMLlibLabeledPoint(boolean includeVariances) {
        this.includeVariances = includeVariances;
    }

    @Override
    public LabeledPoint call(Row inputRow) throws Exception {
        org.apache.spark.mllib.linalg.Vector mllibDenseVector = null;

        double[] averages = ((DenseVector) inputRow.getAs("features")).values();

        if (includeVariances) {
            double[] variances = ((DenseVector) inputRow.getAs("variances")).values();

            double[] features = new double[averages.length + variances.length];
            int i = 0;
            for (Double d: averages) {
                features[i++] = d;
            }

            for (Double d: variances) {
                features[i++] = d;
            }

            mllibDenseVector = Vectors.dense(features);
        } else {
           mllibDenseVector = Vectors.dense(averages);
        }

        return new LabeledPoint(inputRow.getAs("label"), mllibDenseVector);
    }

}
