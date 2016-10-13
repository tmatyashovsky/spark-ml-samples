package com.lohika.morning.ml.spark.driver.service.lyrics;

import java.io.IOException;
import java.util.UUID;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.*;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import static org.apache.spark.sql.functions.column;
import static org.apache.spark.sql.functions.explode;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class Exploder extends Transformer implements MLWritable {

    private static final String ROW_NUMBER = "rowNumber";
    private static final String LABEL = "label";
    private static final String PLURAL_COL = "filteredWords";
    private static final String SINGULAR_COL = "filteredWord";
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
        Column singular = explode(column(PLURAL_COL)).as(SINGULAR_COL);
        return sentences.select(column(ROW_NUMBER), column(LABEL), singular);
    }

    @Override
    public StructType transformSchema(StructType schema) {
        return new StructType(new StructField[]{
                DataTypes.createStructField(ROW_NUMBER, DataTypes.LongType, false),
                DataTypes.createStructField(LABEL, DataTypes.DoubleType, false),
                DataTypes.createStructField(SINGULAR_COL, DataTypes.StringType, false)
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
