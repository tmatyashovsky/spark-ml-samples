package com.lohika.morning.ml.spark.distributed.library.function.map.generic;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Row;

/**
 * Created by tmatyashovsky on 7/28/16.
 */
public class ConvertMLLabeledPointToMLlibLabeledPoint implements Function<Row, LabeledPoint> {

    @Override
    public LabeledPoint call(Row inputRow) throws Exception {
        DenseVector mlDenseVector = (DenseVector) inputRow.getAs("features");
        org.apache.spark.mllib.linalg.Vector mllibDenseVector = Vectors.dense(mlDenseVector.values());
        return new LabeledPoint(inputRow.getAs("label"), mllibDenseVector);
    }

}
