module TensorStream
  ##
  # Utiliity functions for creating performance reports
  class ReportTool
    def self.profile_for(session, order_by: :slowest)
      context = session.last_session_context
      eval_times = context[:profile][:operations].map do |name, profile|
        [name, profile[:eval_time], profile[:tensor].source]
      end

      if order_by == :slowest
        eval_times.sort_by { |a, b| b[1] <=> a[1] }
      else
        eval_times.sort_by { |a, b| a[1] <=> b[1] }
      end
    end
  end
end