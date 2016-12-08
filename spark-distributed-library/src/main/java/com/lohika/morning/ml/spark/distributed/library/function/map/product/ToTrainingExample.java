package com.lohika.morning.ml.spark.distributed.library.function.map.product;

import java.util.stream.IntStream;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;

public class ToTrainingExample implements MapFunction<Row, Row> {

    @Override
    public Row call(final Row inputRow) {
        double[] features = IntStream.rangeClosed(1, 93)
                 .map(i -> inputRow.getAs("feat_" + i))
                 .mapToDouble(i -> i)
                 .toArray();

        return RowFactory.create(inputRow.getAs("target"),
                Vectors.dense(features).toSparse());
    }

}