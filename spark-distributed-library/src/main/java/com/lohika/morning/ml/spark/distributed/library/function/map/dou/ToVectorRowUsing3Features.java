package com.lohika.morning.ml.spark.distributed.library.function.map.dou;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;

public class ToVectorRowUsing3Features  implements Function<Row, Row> {

    @Override
    public Row call(Row inputRow) {
        return RowFactory.create(Vectors.dense(Integer.valueOf(inputRow.getInt(0)).doubleValue(), inputRow.getDouble(1),
                DouConverter.transformEnglishLevel(inputRow.getString(2))));
    }

}