---
title: Advanced SQL Server
date: 2023-07-19 00:00:00 +0800
categories: [SQL, Basic_SQL]
tags: [SQL]
---


# Advanced SQL Server Notes

## Constraints

Constraints are used to specify rules for the data in a table.

### Primary Key Constraint

```sql
ALTER TABLE table
ADD CONSTRAINT pk_constraint PRIMARY KEY (column);
```

### Foreign Key Constraint

```sql
ALTER TABLE table
ADD CONSTRAINT fk_constraint FOREIGN KEY (column)
REFERENCES referenced_table (referenced_column);
```

### Check Constraint

```sql
ALTER TABLE table
ADD CONSTRAINT check_constraint CHECK (condition);
```

## Subqueries

A subquery is a query nested inside another query.

### Scalar Subquery

```sql
SELECT column1, (SELECT column2 FROM table2 WHERE condition) AS subquery_result
FROM table1;
```

### Correlated Subquery

```sql
SELECT column1
FROM table1
WHERE column2 = (SELECT MAX(column2) FROM table1 WHERE condition);
```

## Common Table Expressions (CTE)

A CTE is a named temporary result set.

```sql
WITH cte_name AS (
  SELECT column1, column2
  FROM table
  WHERE condition
)
SELECT * FROM cte_name;
```

## Dynamic SQL

Dynamic SQL allows the construction of SQL statements at runtime.

```sql
DECLARE @sql_query NVARCHAR(MAX);
SET @sql_query = 'SELECT * FROM table WHERE column = ''value''';
EXEC sp_executesql @sql_query;
```

## Window Functions

Window functions operate on a set of rows related to the current row.

### ROW_NUMBER()

Assigns a unique integer to each row within a partition.

```sql
SELECT column1, column2, ROW_NUMBER() OVER (PARTITION BY column1 ORDER BY column2) AS row_num
FROM table;
```

### RANK() and DENSE_RANK()

Assigns a rank to each row based on the values in the ORDER BY clause.

```sql
SELECT column1, column2, RANK() OVER (ORDER BY column2) AS rank
FROM table;
```

### LEAD() and LAG()

Access data from subsequent or previous rows in the result set.

```sql
SELECT column1, LAG(column1) OVER (ORDER BY column2) AS previous_value
FROM table;
```

## Full-Text Search

Full-Text Search enables fast and flexible indexing for keyword-based searches.

### Creating a Full-Text Catalog

```sql
CREATE FULLTEXT CATALOG catalog_name AS DEFAULT;
```

### Full-Text Search Query

```sql
SELECT column1, column2
FROM table
WHERE CONTAINS(column1, 'search_term');
```

## JSON Support

SQL Server has native support for handling JSON data.

### Storing JSON Data

```sql
INSERT INTO table (json_column)
VALUES ('{"key": "value"}');
```

### Querying JSON Data

```sql
SELECT json_column->'key' AS value
FROM table;
```

## Spatial Data

SQL Server supports spatial data types and operations.

### Creating a Spatial Table

```sql
CREATE TABLE spatial_table (id INT PRIMARY KEY, location GEOMETRY);
```

### Spatial Query

```sql
SELECT id
FROM spatial_table
WHERE location.STDistance(geometry::STGeomFromText('POINT(1 2)', 0)) < 10;
```
