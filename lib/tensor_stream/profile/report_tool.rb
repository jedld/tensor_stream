module TensorStream
  ##
  # Utiliity functions for creating performance reports
  class ReportTool
    def self.profile_for(session, order_by: :slowest)
      context = session.last_session_context
      eval_times = context[:profile][:operations].map { |name, profile|
        [name, profile[:op], profile[:eval_time], profile[:shape]]
      }

      if order_by == :slowest
        eval_times.sort_by { |a| a[2] }.reverse!
      else
        eval_times.sort_by { |a| a[2] }
      end
    end
  end
end
