package com.lohika.morning.ml.spark.distributed.library.function.map.generic;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Row;

public class MapParquetToLabeledPoint implements Function<Row, LabeledPoint> {

    @Override
    public LabeledPoint call(Row inputRow) {
        return new LabeledPoint(inputRow.getDouble(1), (Vector) inputRow.get(0));
    }

}