
"""
NeuroWorld AI
File: backend/app/database/repositories/base.py

Generic SQLAlchemy 2.x repository.

Features:
- Create, read, update, delete
- Pagination and filtering
- Record counting
- Transaction support
- Optional row locking
- Safe model-field validation
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generic, Iterator, TypeVar

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import Session

ModelT = TypeVar("ModelT")


class BaseRepository(Generic[ModelT]):
    def __init__(
        self,
        session: Session,
        model: type[ModelT],
    ) -> None:
        self.session = session
        self.model = model

        mapper = inspect(model)
        self._columns = {
            column.key
            for column in mapper.columns
        }

    def _validate_fields(
        self,
        fields: dict[str, Any],
    ) -> None:
        unknown = set(fields) - self._columns

        if unknown:
            raise ValueError(
                "Unknown model fields: "
                + ", ".join(sorted(unknown))
            )

    def _commit(self) -> None:
        try:
            self.session.commit()
        except SQLAlchemyError:
            self.session.rollback()
            raise

    @contextmanager
    def transaction(self) -> Iterator[Session]:
        """
        Transaction context.

        Use with a session that is not already inside
        an active transaction.
        """
        with self.session.begin():
            yield self.session

    def create(
        self,
        *,
        commit: bool = True,
        **fields: Any,
    ) -> ModelT:
        self._validate_fields(fields)

        instance = self.model(**fields)

        self.session.add(instance)

        if commit:
            self._commit()
            self.session.refresh(instance)
        else:
            self.session.flush()

        return instance

    def get(
        self,
        record_id: Any,
    ) -> ModelT | None:
        return self.session.get(
            self.model,
            record_id,
        )

    def get_or_raise(
        self,
        record_id: Any,
    ) -> ModelT:
        instance = self.get(record_id)

        if instance is None:
            raise LookupError(
                f"{self.model.__name__} "
                f"{record_id!r} not found"
            )

        return instance

    def get_by(
        self,
        **filters: Any,
    ) -> ModelT | None:
        self._validate_fields(filters)

        statement = (
            select(self.model)
            .filter_by(**filters)
            .limit(1)
        )

        return self.session.scalar(statement)

    def list(
        self,
        *,
        skip: int = 0,
        limit: int = 100,
        order_by: str = "id",
        descending: bool = True,
        **filters: Any,
    ) -> list[ModelT]:
        if skip < 0:
            raise ValueError(
                "skip must be nonnegative"
            )

        if not 1 <= limit <= 1000:
            raise ValueError(
                "limit must be between 1 and 1000"
            )

        self._validate_fields(filters)

        if order_by not in self._columns:
            raise ValueError(
                f"Invalid ordering field: {order_by}"
            )

        order_column = getattr(
            self.model,
            order_by,
        )

        statement = (
            select(self.model)
            .filter_by(**filters)
            .order_by(
                order_column.desc()
                if descending
                else order_column.asc()
            )
            .offset(skip)
            .limit(limit)
        )

        return list(
            self.session.scalars(
                statement
            ).all()
        )

    def count(
        self,
        **filters: Any,
    ) -> int:
        self._validate_fields(filters)

        statement = (
            select(func.count())
            .select_from(self.model)
            .filter_by(**filters)
        )

        return int(
            self.session.scalar(statement) or 0
        )

    def exists(
        self,
        **filters: Any,
    ) -> bool:
        return self.get_by(**filters) is not None

    def update(
        self,
        record_id: Any,
        *,
        commit: bool = True,
        **fields: Any,
    ) -> ModelT | None:
        self._validate_fields(fields)

        instance = self.get(record_id)

        if instance is None:
            return None

        for name, value in fields.items():
            setattr(
                instance,
                name,
                value,
            )

        self.session.add(instance)

        if commit:
            self._commit()
            self.session.refresh(instance)
        else:
            self.session.flush()

        return instance

    def delete(
        self,
        record_id: Any,
        *,
        commit: bool = True,
    ) -> bool:
        instance = self.get(record_id)

        if instance is None:
            return False

        self.session.delete(instance)

        if commit:
            self._commit()
        else:
            self.session.flush()

        return True

    def delete_by(
        self,
        *,
        commit: bool = True,
        **filters: Any,
    ) -> int:
        self._validate_fields(filters)

        statement = (
            select(self.model)
            .filter_by(**filters)
        )

        records = list(
            self.session.scalars(
                statement
            ).all()
        )

        for record in records:
            self.session.delete(record)

        if commit:
            self._commit()
        else:
            self.session.flush()

        return len(records)

    def get_for_update(
        self,
        record_id: Any,
    ) -> ModelT | None:
        """
        Lock a row where supported.

        SQLite does not provide equivalent
        SELECT FOR UPDATE row locking.
        """
        primary_key = inspect(
            self.model
        ).primary_key

        if len(primary_key) != 1:
            raise ValueError(
                "Single-column primary key required"
            )

        statement = (
            select(self.model)
            .where(
                primary_key[0] == record_id
            )
            .with_for_update()
        )

        return self.session.scalar(
            statement
        )
