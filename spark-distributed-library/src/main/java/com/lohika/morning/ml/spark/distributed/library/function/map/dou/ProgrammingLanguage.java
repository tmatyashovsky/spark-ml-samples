package com.lohika.morning.ml.spark.distributed.library.function.map.dou;

public enum ProgrammingLanguage {

    oneC("1C", 0),
    apl("APL", 1),
    c("C", 2),
    cSharp("C#/.NET", 3),
    cPlusPlus("C++", 4),
    delphi("Delphi", 5),
    flex("Flex/Flash/AIR", 6),
    haskell("Haskell", 7),
    java("Java", 8),
    javaScript("JavaScript", 9),
    objectiveC("Objective-C", 10),
    other("Other", 11),
    perl("Perl", 12),
    php("PHP", 13),
    python("Python", 14),
    ruby("Ruby/Rails", 15),
    scala("Scala", 16),
    sql("SQL", 17),
    swift("Swift", 18),
    notDefined("", -42);

    private String name;
    private Integer value;

    ProgrammingLanguage(String name, Integer value) {
        this.name = name;
        this.value = value;
    }

    public String getName() {
        return name;
    }

    public Integer getValue() {
        return value;
    }
}
