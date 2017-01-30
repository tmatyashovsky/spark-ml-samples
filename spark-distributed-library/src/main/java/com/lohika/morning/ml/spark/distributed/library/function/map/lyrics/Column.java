package com.lohika.morning.ml.spark.distributed.library.function.map.lyrics;

import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;

public enum Column {

    VALUE("value", DataTypes.StringType),

    CLEAN("clean", DataTypes.StringType),

    ID("id", DataTypes.StringType),

    ROW_NUMBER("rowNumber", DataTypes.IntegerType),

    WORDS("words", DataTypes.createArrayType(DataTypes.StringType)),

    LABEL("label", DataTypes.DoubleType),

    FILTERED_WORD("filteredWord", DataTypes.StringType),

    FILTERED_WORDS("filteredWords", DataTypes.createArrayType(DataTypes.StringType)),

    STEMMED_WORD("stemmedWord", DataTypes.StringType),

    STEMMED_SENTENCE("stemmedSentence", DataTypes.StringType),

    VERSE("verse", DataTypes.createArrayType(DataTypes.StringType));

    private final String name;
    private final DataType dataType;

    Column(final String name, final DataType dataType) {
        this.name = name;
        this.dataType = dataType;
    }

    public StructField getStructType() {
        return DataTypes.createStructField(name, dataType, false);
    }

    public String getName() {
        return name;
    }
}
