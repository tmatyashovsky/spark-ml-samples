package com.lohika.morning.ml.spark.driver.service.lyrics.transformer;

import com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column;
import java.io.IOException;
import java.util.UUID;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import static org.apache.spark.sql.functions.column;
import static org.apache.spark.sql.functions.explode;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class Exploder extends Transformer implements MLWritable {

    private String uid;

    public Exploder(String uid) {
        this.uid = uid;
    }

    public Exploder() {
        this.uid = "Exploder" + "_" + UUID.randomUUID().toString();
    }

    @Override
    public Dataset<Row> transform(Dataset<?> sentences) {
        // Create as many rows as elements in provided column.
        org.apache.spark.sql.Column singular = explode(column(Column.FILTERED_WORDS.getName())).as(Column.FILTERED_WORD.getName());
        return sentences.select(column(Column.ID.getName()), column(Column.ROW_NUMBER.getName()), column(Column.LABEL.getName()), singular);
    }

    @Override
    public StructType transformSchema(StructType schema) {
        return new StructType(new StructField[]{
                Column.ID.getStructType(),
                Column.ROW_NUMBER.getStructType(),
                Column.LABEL.getStructType(),
                Column.FILTERED_WORD.getStructType()
        });
    }

    @Override
    public Transformer copy(ParamMap extra) {
        return super.defaultCopy(extra);
    }

    @Override
    public String uid() {
        return this.uid;
    }

    @Override
    public MLWriter write() {
        return new DefaultParamsWriter(this);
    }

    @Override
    public void save(String path) throws IOException {
        write().save(path);
    }

    public static MLReader<Exploder> read() {
        return new DefaultParamsReader<>();
    }

}
