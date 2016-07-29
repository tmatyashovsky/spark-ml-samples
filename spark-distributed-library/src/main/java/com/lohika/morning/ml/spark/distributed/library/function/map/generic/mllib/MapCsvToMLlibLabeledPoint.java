package com.lohika.morning.ml.spark.distributed.library.function.map.generic.mllib;

import java.util.stream.IntStream;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Row;

public class MapCsvToMLlibLabeledPoint implements Function<Row, LabeledPoint> {

    @Override
    public LabeledPoint call(Row inputRow) {
        double[] features = new double[inputRow.size() - 1];

        IntStream.range(1, inputRow.size()).forEach(i -> features[i - 1] = inputRow.getInt(i));

        return new LabeledPoint(Integer.valueOf(inputRow.getInt(0)).doubleValue(), Vectors.dense(features));
    }

}