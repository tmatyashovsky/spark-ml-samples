package com.lohika.morning.ml.spark.distributed.library.function.map.generic;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Row;

public class ConvertMLLabeledPointToMLlibLabeledPoint implements Function<Row, LabeledPoint> {

    @Override
    public LabeledPoint call(Row inputRow) throws Exception {
        double[] averages = ((DenseVector) inputRow.getAs("averages")).values();
        double[] variances = ((DenseVector) inputRow.getAs("variances")).values();

        double[] features = new double[averages.length + variances.length];
        int i = 0;
        for (Double d: averages) {
            features[i++] = d;
        }

        for (Double d: variances) {
            features[i++] = d;
        }

        org.apache.spark.mllib.linalg.Vector mllibDenseVector = Vectors.dense(features);
        return new LabeledPoint(inputRow.getAs("label"), mllibDenseVector);
//        DenseVector mlDenseVector = (DenseVector) inputRow.getAs("features");
//        org.apache.spark.mllib.linalg.Vector mllibDenseVector = Vectors.dense(mlDenseVector.values());
//        return new LabeledPoint(inputRow.getAs("label"), mllibDenseVector);
    }

}
