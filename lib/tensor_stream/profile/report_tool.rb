module TensorStream
  ##
  # Utiliity functions for creating performance reports
  class ReportTool
    def self.profile_for(session, order_by: :slowest)
      context = session.last_session_context
      eval_times = context[:profile][:operations].map do |name, profile|
        [name, profile[:eval_time], profile[:shape], profile[:tensor].source]
      end

      if order_by == :slowest
        eval_times.sort_by { |a| a[1] }.reverse!
      else
        eval_times.sort_by { |a| a[1] }
      end
    end
  end
end