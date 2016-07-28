package com.lohika.morning.ml.spark.distributed.library.function.map;

import java.util.Collections;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;

/**
 * Created by tmatyashovsky on 7/28/16.
 */
public class Map implements Function<Row, Row> {
    @Override
    public Row call(Row v1) throws Exception {
        return RowFactory.create(Collections.singletonList(v1.getString(0)));
    }
}
