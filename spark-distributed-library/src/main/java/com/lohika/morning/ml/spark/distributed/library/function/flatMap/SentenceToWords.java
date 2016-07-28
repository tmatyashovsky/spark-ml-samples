package com.lohika.morning.ml.spark.distributed.library.function.flatMap;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;

/**
 * Created by tmatyashovsky on 7/28/16.
 */
public class SentenceToWords implements FlatMapFunction<Row, Row> {

    @Override
    public Iterator<Row> call(Row row) throws Exception {
        List<Row> rows = new ArrayList<Row>();

        row.getList(2).stream().forEach(x -> rows.add(RowFactory.create(x)));

        return rows.iterator();
    }

}
