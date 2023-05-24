from datetime import datetime, timezone, timedelta
from google.protobuf.timestamp_pb2 import Timestamp

from nbox.hyperloop.jobs.job_pb2 import Job as JobProto

class Schedule:
  def __init__(
    self,
    hour: int  = None,
    minute: int = None,
    days: list = [],
    months: list = [],
    starts: datetime = None,
    ends: datetime = None,
  ):
    """Make scheduling natural. Uses 24-hour nomenclature.

    Args:
      hour (int): Hour of the day, if only this value is passed it will run every `hour`
      minute (int): Minute of the hour, if only this value is passed it will run every `minute`
      days (str/list, optional): List of days (first three chars) of the week, if not passed it will run every day.
      months (str/list, optional): List of months (first three chars) of the year, if not passed it will run every month.
      starts (datetime, optional): UTC Start time of the schedule, if not passed it will start now.
      ends (datetime, optional): UTC End time of the schedule, if not passed it will end in 7 days.

    Examples:

      # 4:20PM everyday
      Schedule(16, 20)

      # 4:20AM every friday
      Schedule(4, 20, ["fri"])

      # 4:20AM every friday from jan to feb
      Schedule(4, 20, ["fri"], ["jan", "feb"])

      # 4:20PM everyday starting in 2 days and runs for 3 days
      starts = datetime.now(timezone.utc) + timedelta(days = 2) # NOTE: that time is in UTC
      Schedule(16, 20, starts = starts, ends = starts + timedelta(days = 3))

      # Every 1 hour
      Schedule(1)

      # Every 69 minutes
      Schedule(minute = 69)
    """
    self.hour = hour
    self.minute = minute

    self._is_repeating = self.hour or self.minute
    self.mode = None
    if self.hour == None and self.minute == None:
      raise ValueError("Atleast one of hour or minute should be passed")
    elif self.hour != None and self.minute != None:
      assert self.hour in list(range(0, 24)), f"Hour must be in range 0-23, got {self.hour}"
      assert self.minute in list(range(0, 60)), f"Minute must be in range 0-59, got {self.minute}"
    elif self.hour != None:
      assert self.hour in list(range(0, 24)), f"Hour must be in range 0-23, got {self.hour}"
      self.mode = "every"
      self.minute = datetime.now(timezone.utc).strftime("%m") # run every this minute past this hour
      self.hour = f"*/{self.hour}"
    elif self.minute != None:
      self.hour = self.minute // 60
      assert self.hour in list(range(0, 24)), f"Hour must be in range 0-23, got {self.hour}"
      self.minute = f"*/{self.minute % 60}"
      self.hour = f"*/{self.hour}" if self.hour > 0 else "*"
      self.mode = "every"

    _days = {k:str(i) for i,k in enumerate(["sun","mon","tue","wed","thu","fri","sat"])}
    _months = {k:str(i+1) for i,k in enumerate(["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])}

    if isinstance(days, str):
      days = [days]
    if isinstance(months, str):
      months = [months]

    diff = set(days) - set(_days.keys())
    if len(diff):
      raise ValueError(f"Invalid days: {diff}")
    self.days = ",".join([_days[d] for d in days]) if days else "*"

    diff = set(months) - set(_months.keys())
    if len(diff):
      raise ValueError(f"Invalid months: {diff}")
    self.months = ",".join([_months[m] for m in months]) if months else "*"

    self.starts = starts or datetime.now(timezone.utc)
    self.ends = ends or datetime.now(timezone.utc) + timedelta(days = 7)

  @property
  def cron(self):
    """Get the cron string for the given schedule"""
    if self.mode == "every":
      return f"{self.minute} {self.hour} * * *"
    return f"{self.minute} {self.hour} * {self.months} {self.days}"

  def get_dict(self):
    """Get the dictionary representation of this Schedule"""
    return {"cron": self.cron, "mode": self.mode, "starts": self.starts, "ends": self.ends}

  def get_message(self) -> JobProto.Schedule:
    """Get the JobProto.Schedule object for this Schedule"""
    _starts = Timestamp()
    _starts.GetCurrentTime()
    _ends = Timestamp()
    _ends.FromDatetime(self.ends)
    return JobProto.Schedule(
      start = _starts,
      end = _ends,
      cron = self.cron
    )

  def __repr__(self):
    return str(self.get_dict())