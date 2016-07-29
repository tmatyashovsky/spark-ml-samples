package com.lohika.morning.ml.spark.distributed.library.function.map.generic.ml;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.sql.Row;

public class MapParquetToLabeledPoint implements Function<Row, LabeledPoint> {

    @Override
    public LabeledPoint call(Row inputRow) {
        return new LabeledPoint(inputRow.getAs("label"), inputRow.getAs("features"));
    }

}