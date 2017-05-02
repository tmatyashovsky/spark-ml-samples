package com.lohika.morning.ml.spark.distributed.library.function.map.generic.mllib;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import static org.apache.spark.mllib.linalg.Vectors.dense;
import org.apache.spark.sql.Row;

public class MapRowToMLlibVector implements Function<Row, Vector> {

    @Override
    public Vector call(Row inputRow) throws Exception {
        final DenseVector mlDenseVector = inputRow.getAs("features");
        return dense(mlDenseVector.toArray());
    }

}