---
title: Advanced SQL Server
date: 2023-07-19 00:00:00 +0800
categories: [SQL, Basic_SQL]
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
