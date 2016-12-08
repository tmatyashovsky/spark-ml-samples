package com.lohika.morning.ml.spark.driver.service.lyrics.transformer;

import com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column;
import com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.StemmingFunction;
import java.io.IOException;
import java.util.UUID;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class Stemmer extends Transformer implements MLWritable {

    private String uid;

    public Stemmer(String uid) {
        this.uid = uid;
    }

    public Stemmer() {
        this.uid = "CustomStemmer" + "_" + UUID.randomUUID().toString();
    }

    @Override
    public String uid() {
        return this.uid;
    }

    @Override
    public Dataset<Row> transform(Dataset dataset) {
        return dataset.map(new StemmingFunction(), RowEncoder.apply(this.transformSchema(dataset.schema())));
    }

    @Override
    public StructType transformSchema(StructType schema) {
        return new StructType(new StructField[]{
                Column.ID.getStructType(),
                Column.ROW_NUMBER.getStructType(),
                Column.LABEL.getStructType(),
                Column.STEMMED_WORD.getStructType()
        });
    }

    @Override
    public Transformer copy(ParamMap extra) {
        return super.defaultCopy(extra);
    }

    @Override
    public MLWriter write() {
        return new DefaultParamsWriter(this);
    }

    @Override
    public void save(String path) throws IOException {
        write().save(path);
    }

    public static MLReader<Stemmer> read() {
        return new DefaultParamsReader<>();
    }

}
