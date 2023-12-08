---
title: Advanced SQL Server
date: 2023-09-05 00:00:00 +0800
categories: [SQL, SQL_Server]
tags: [SQL]
---

# Advanced SQL Server 

## Indexing

### Creating an Index

Indexes improve the speed of data retrieval operations on a table.

```sql
CREATE INDEX index_name
ON table (column1, column2);
```

### Removing an Index

```sql
DROP INDEX index_name
ON table;
```

## Stored Procedures

A stored procedure is a precompiled collection of one or more SQL statements.

### Creating a Stored Procedure

```sql
CREATE PROCEDURE procedure_name
AS
BEGIN
  -- SQL statements
END;
```

### Executing a Stored Procedure

```sql
EXEC procedure_name;
```

## Triggers

A trigger is a set of instructions that are automatically executed, or "triggered," in response to certain events.

### Creating a Trigger

```sql
CREATE TRIGGER trigger_name
ON table
AFTER INSERT, UPDATE, DELETE
AS
BEGIN
  -- Trigger logic
END;
```

## Transactions

A transaction is a sequence of one or more SQL statements that are executed as a single unit of work.

### Starting a Transaction

```sql
BEGIN TRANSACTION;
```

### Committing a Transaction

```sql
COMMIT;
```

### Rolling Back a Transaction

```sql
ROLLBACK;
```

## Views

A view is a virtual table based on the result of a SELECT query.

### Creating a View

```sql
CREATE VIEW view_name AS
SELECT column1, column2
FROM table
WHERE condition;
```

### Altering a View

```sql
ALTER VIEW view_name AS
SELECT column1, column2
FROM table
WHERE condition;
```

### Dropping a View

```sql
DROP VIEW view_name;
```

### Creating a Materialized View

```sql
CREATE MATERIALIZED VIEW view_name AS
SELECT column1, column2
FROM table
WHERE condition;
```

### Refreshing a Materialized View

```sql
REFRESH MATERIALIZED VIEW view_name;
```

### Dropping a Materialized View

```sql
DROP MATERIALIZED VIEW view_name;
```

### Creating an Updatable View

```sql
CREATE VIEW view_name AS
SELECT column1, column2
FROM table
WHERE condition
WITH CHECK OPTION;
```

### Creating an Indexed View

```sql
CREATE VIEW view_name WITH SCHEMABINDING AS
SELECT column1, column2
FROM table
WHERE condition;
CREATE UNIQUE CLUSTERED INDEX idx_indexed_view ON view_name (column1);
```

## User-Defined Functions

A user-defined function is a set of SQL statements that returns a value.

### Creating a Scalar Function

```sql
CREATE FUNCTION function_name (@parameter1 INT, @parameter2 INT)
RETURNS INT
AS
BEGIN
  -- Function logic
END;
```

### Creating a Table-Valued Function

```sql
CREATE FUNCTION function_name (@parameter1 INT, @parameter2 INT)
RETURNS TABLE
AS
RETURN
  SELECT column1, column2
  FROM table
  WHERE condition;
```

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
